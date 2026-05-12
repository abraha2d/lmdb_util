#!/usr/bin/env python3

import argparse
import logging
from pprint import pprint
from pathlib import Path
import struct

logger = logging.getLogger("lmdb_util")


class LogFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + "[DBUG] " + format_str + reset,
        logging.INFO: grey + "[INFO] " + format_str + reset,
        logging.WARNING: yellow + "[WARN] " + format_str + reset,
        logging.ERROR: red + "[ERR ] " + format_str + reset,
        logging.CRITICAL: bold_red + "[CRIT] " + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class MDBDB:
    """Information about a single database in the environment."""

    # uint32_t  pad
    # uint16_t  flags
    # uint16_t  depth
    # size_t    branch_pages
    # size_t    leaf_pages
    # size_t    overflow_pages
    # size_t    entries
    # size_t    root
    _format = '<4xHHQQQQQ'
    _size = struct.calcsize(_format)

    def __init__(self, data: memoryview):
        self.flags, self.depth, self.branch_pages, self.leaf_pages, self.overflow_pages, self.entries, self.root = struct.unpack(
            self._format, data)

    @property
    def root_page(self):
        return MDBPage.get(self.root)


class MDBNode:
    """A single key/data pair within a page."""

    # uint16_t  lo
    # uint16_t  hi
    # uint16_t  flags
    # uint16_t  ksize
    _header_format = '<HHHH'
    _header_size = struct.calcsize(_header_format)

    def __init__(self, page: memoryview, ptr: int):
        self._page = page
        self._ptr = ptr
        self.lo, self.hi, self.flags, self.ksize = struct.unpack(self._header_format, page[ptr:ptr + self._header_size])

    @property
    def key(self):
        key_start = self._ptr + self._header_size
        key_end = key_start + self.ksize
        return self._page[key_start:key_end].tobytes()


class MDBBranchNode(MDBNode):
    """
    A branch node.

    `lo`, `hi`, and `flags` are used for child pgno.
    """

    @property
    def pgno(self):
        return self.lo | (self.hi << 16) | (self.flags << 32)

    @property
    def page(self):
        return MDBPage.get(self.pgno)


class MDBLeafNode(MDBNode):
    """
    A leaf node.

    `lo` and `hi` are used for data size.
    `flags` describe node contents.
    """

    F_BIGDATA = 0x01  # `data` is the page number of an overflow page with actual data
    F_SUBDATA = 0x02  # `data` is a sub-database
    F_DUPDATA = 0x04  # `data` has duplicates

    @property
    def data(self):
        data_start = self._ptr + self._header_size + self.ksize

        if self.flags & self.F_BIGDATA:
            data_format = '<Q'
            data_size = struct.calcsize(data_format)
            data_end = data_start + data_size

            pgno, = struct.unpack(data_format, self._page[data_start:data_end])
            page = MDBPage.get(pgno)
            assert page.__class__ == OverflowPage, "Expected overflow page"
            page: OverflowPage
            data = page.data

        else:
            data_size = self.lo | (self.hi << 16)
            data_end = data_start + data_size
            data = self._page[data_start:data_end]

        if self.flags & self.F_SUBDATA:
            return MDBDB(data)

        return data


class MDBPage:
    """Base class for all page types."""

    PAGE_SIZE = 4096

    P_BRANCH = 0x01
    P_LEAF = 0x02
    P_OVERFLOW = 0x04
    P_META = 0x08

    # size_t    mp_pgno
    # uint16_t  mp_pad
    # uint16_t  mp_flags
    # uint16_t  mp_lower
    # uint16_t  mp_upper
    _header_format = '<Q2xHHH'
    _header_size = struct.calcsize(_header_format)

    _db: memoryview = None
    _pages: dict[int, 'MDBPage'] = {}

    def __init__(self, pgno: int):
        assert self._db is not None, "DB memoryview not initialized"
        self.pgno = pgno

        pgno, flags, lower, upper = struct.unpack(self._header_format, self._data[:self._header_size])

        if pgno != self.pgno:
            logger.warning(f"Page {self.pgno} seems corrupted (pgno does not match). Ignoring...")

        self.flags: int = flags
        self.lower: int = lower
        self.upper: int = upper

        MDBPage._pages[pgno] = self

    @property
    def _data(self):
        return self._db[self.pgno * self.PAGE_SIZE:(self.pgno + 1) * self.PAGE_SIZE]

    @staticmethod
    def get(pgno: int):
        if pgno in MDBPage._pages:
            return MDBPage._pages[pgno]

        page = MDBPage(pgno)

        if page.flags & MDBPage.P_META:
            MetaPage._init_meta(page)
        elif page.flags & MDBPage.P_BRANCH:
            BranchPage._init_branch(page)
        elif page.flags & MDBPage.P_LEAF:
            LeafPage._init_leaf(page)
        elif page.flags & MDBPage.P_OVERFLOW:
            OverflowPage._init_overflow(page)

        return page


class MetaPage(MDBPage):
    """The start point for accessing a database snapshot."""

    # uint32_t  magic
    # uint32_t  version
    # void*     address
    # size_t    mapsize
    _prefix_format = '<LLQQ'
    _prefix_size = struct.calcsize(_prefix_format)

    # size_t    last_pg
    # size_t    txnid
    _suffix_format = '<QQ'
    _suffix_size = struct.calcsize(_suffix_format)

    def _init_meta(self: MDBPage):
        self.__class__ = MetaPage
        self: MetaPage

        prefix_start = self._header_size
        prefix_end = self._header_size + self._prefix_size
        prefix_data = self._data[prefix_start:prefix_end]
        self.magic, self.version, self.address, self.mapsize = struct.unpack('<IIQQ', prefix_data)

        free_db_start = prefix_end
        free_db_end = free_db_start + MDBDB._size
        self.free_db = MDBDB(self._data[free_db_start:free_db_end])

        main_db_start = free_db_end
        main_db_end = main_db_start + MDBDB._size
        self.main_db = MDBDB(self._data[main_db_start:main_db_end])

        suffix_start = main_db_end
        suffix_end = suffix_start + self._suffix_size
        self.last_pg, self.txnid = struct.unpack(self._suffix_format, self._data[suffix_start:suffix_end])


class TreePage(MDBPage):
    """A branch or leaf page."""

    nodes: list[MDBNode]

    def _init_tree(self: MDBPage):
        self.__class__ = TreePage
        num_nodes = (self.lower - self._header_size) // 2

        ptrs_format = f'<{num_nodes}H'
        ptrs_size = struct.calcsize(ptrs_format)

        ptrs_start = self._header_size
        ptrs_end = ptrs_start + ptrs_size
        ptrs = struct.unpack(ptrs_format, self._data[ptrs_start:ptrs_end])

        self.nodes = [MDBNode(self._data, ptr) for ptr in ptrs]


class BranchPage(TreePage):
    """A branch page."""

    nodes: list[MDBBranchNode]

    def _init_branch(self: MDBPage):
        TreePage._init_tree(self)

        self.__class__ = BranchPage
        self: BranchPage

        for node in self.nodes:
            node.__class__ = MDBBranchNode


class LeafPage(TreePage):
    """A leaf page."""

    nodes: list[MDBLeafNode]

    def _init_leaf(self: MDBPage):
        TreePage._init_tree(self)

        self.__class__ = LeafPage
        self: LeafPage

        for node in self.nodes:
            node.__class__ = MDBLeafNode


class OverflowPage(MDBPage):
    """An overflow page."""

    def _init_overflow(self: MDBPage):
        self.__class__ = OverflowPage
        self: OverflowPage

    @property
    def pages(self) -> int:
        return self.lower | (self.upper << 16)

    @property
    def data(self) -> memoryview:
        data_start = self.pgno * self.PAGE_SIZE + self._header_size
        data_size = self.pages * self.PAGE_SIZE
        data_end = data_start + data_size
        return self._db[data_start:data_end]


def walk_keys(page: MDBPage, parents=''):
    """Returns a depth-first list of keys."""
    logger.debug(f"Scanning page {parents}{page.pgno}")
    keys = []

    if type(page) == BranchPage:
        page: BranchPage
        for node in page.nodes:
            keys.extend(walk_keys(node.page, f"{parents}{page.pgno}->"))
        return keys

    elif type(page) == LeafPage:
        page: LeafPage
        keys = [node.key for node in page.nodes]

    else:
        logger.warning(f"Got unexpected type {page.flags} for page {page.pgno} ({type(page).__name__}). Skipping...")

    return keys


def run_list(args):
    """List subdatabase names."""

    meta0: MetaPage = MDBPage.get(0)  # type: ignore
    logger.debug(f"Meta 0: txnid {meta0.txnid}")
    meta1: MetaPage = MDBPage.get(1)  # type: ignore
    logger.debug(f"Meta 1: txnid {meta1.txnid}")

    if args.use_old_meta:
        logger.info("Using older meta page")
        meta = meta0 if meta0.txnid < meta1.txnid else meta1
    else:
        meta = meta0 if meta0.txnid > meta1.txnid else meta1

    root_page = meta.main_db.root_page
    keys = walk_keys(root_page)

    for key in keys:
        print(key.decode())


def setup_cli():
    """Sets up the parser with subcommands."""
    parser = argparse.ArgumentParser(description="Utilities for recovering damaged LMDB data files.")

    subparsers = parser.add_subparsers(
        title='Available commands',
        dest='command',
        required=True,
    )

    parser_list = subparsers.add_parser('list',
                                        help='List subdatabase names.')

    parser.add_argument(
        'data_mdb',
        type=str,
        help='The path to the LMDB data file.'
    )
    parser.add_argument('-v', '--verbose', help="Enable verbose logging.", action="store_const", dest="loglevel",
                        const=logging.DEBUG)
    parser.add_argument('-p', '--use-old-meta', help="Use the older meta page of the two.", action='store_true')

    return parser


def main():
    parser = setup_cli()
    args = parser.parse_args()

    if args.loglevel:
        logger.setLevel(args.loglevel)

    handler = logging.StreamHandler()
    handler.setFormatter(LogFormatter())
    logger.addHandler(handler)

    data = Path(args.data_mdb).read_bytes()
    MDBPage._db = memoryview(data)

    if args.command == 'list':
        run_list(args)


if __name__ == "__main__":
    main()
