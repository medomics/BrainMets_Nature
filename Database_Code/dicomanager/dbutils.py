import sqlite3
sqlite3.connect(":memory:").close()  # Fixes for MacOS
from utils import unnest
import utils
import numpy as np
import re
import ast
import contextlib


# Converts np.array to TEXT when inserting
def adapt_array(arr):
    return np.array2string(arr, separator=',', suppress_small=True)

sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
def convert_array(text):
    decode = text.decode('utf-8')
    pattern = re.compile("^[\[\] 0-9,.-]*$")  # check in basic format
    if pattern.match(decode.replace('\n', '')):
        return np.array(ast.literal_eval(decode)).astype(np.float32)
    return decode

sqlite3.register_converter('ARRAY', convert_array)

def adapt_name_list(names):
    if names == 'NULL':
        return 'NULL'
    return ','.join([str(x) for x in names])

sqlite3.register_adapter(list, adapt_name_list)

def convert_name_list(text):
    if text == b'NULL':
        return text
    return text.decode('utf-8').split(',')

sqlite3.register_converter('NAMELIST', convert_name_list)

def make_connection(db_path, dicts=False):
    """Makes a connection to the database at db_path

    Args:
        db_path (str): A Posix path or :memory:

    Returns:
        sqlite3.Connection: A connection to the database

    Notes:
        Uses PRAGMA synchronous = Extra, journal_mode = WAL
        Does not check for same thread, allowing for multithreading
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute('''PRAGMA synchronous = OFF''')
    conn.execute('''PRAGMA journal_mode = MEMORY''')
    if dicts:
        conn.row_factory = dict_factory
    return conn


def query_db(db_name: str, query: str, dicts: bool = False) -> list:
    """Handles queries to the database cleanly

    Args:
        db_name (str): Database name or path
        query (str): Formatted SQLite3 query
        dicts (bool, optional): Specifies if results are returned as dict (True)
            or as lists (False). Defaults to False.

    Returns:
        list: A list of the query results
    """
    with contextlib.closing(make_connection(db_name, dicts)) as conn:
        with conn:
            return conn.execute(query).fetchall()


def initialize_db(db_path, config_file='table_config.yaml'):
    """Initializes database at db_path as described in tables

    Args:
        config_file (str): A string to the config path.
        db_path (str): Posix path or :memory:

    Returns:
        sqlite3.Connection: Returns a sqlite3 connection to the database
    """
    tables = utils.read_yaml(config_file)
    with make_connection(db_path) as conn:
        for table in tables.keys():
            field_set = [f"'{col}' {fmt}" for col, fmt in tables[table].items()]

            if len(field_set):
                field_fmt = ", ".join(field_set)
                query = f"CREATE TABLE IF NOT EXISTS {table} ({field_fmt})"
                conn.execute(query)
        return conn


def list_columns(conn, table):
    c = conn.execute(f'SELECT * FROM {table}')
    columns = list(map(lambda x: x[0], c.description))
    return columns


def add_rows(conn: sqlite3.Connection, table: str, rows: list) -> None:
    """Adds rows to table with connection to the database

    Args:
        conn (sqlite3.Connection): Connection to the database
        table (str): Table to add rows to database
        rows (list): Rows to add to table
    """
    if isinstance(rows, dict):
        rows = [rows]
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            rows[i] = list(r.values())

    with conn:
        if len(rows):
            qmarks = "?" + ",?" * (len(rows[0]) - 1)

            try:
                conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({qmarks})", rows)
            except Exception:
                print(f'Could not add {len(rows)} rows to {table}. Example row {rows[0]}')
                raise sqlite3.OperationalError


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class GroupingIterator:
    def __init__(self, db_name, table, column):
        self.conn = make_connection(db_name)
        self.conn.row_factory = dict_factory
        self.db_name = db_name
        self.table = table
        self.column = column
        self.unique_values = self._uniques()
        self._len = len(self.unique_values)
        self._index = 0

    def _uniques(self):
        c = self.conn.execute(f'SELECT DISTINCT({self.column}) FROM {self.table}')
        return [x[self.column] for x in c.fetchall()]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._len:
            current = self.unique_values[self._index]
            self._index += 1
            query = f'SELECT * FROM {self.table} WHERE {self.column} = "{current}"'
            try:
                c = self.conn.execute(query)
            except Exception as exp:
                print(query)
                raise exp
            else:
                return c.fetchall()
        raise StopIteration


def sql_wal_compensator(fn):
    def wrapped(cls, *args, **kwargs):
        # Unlocks file if on disk, writes to DB if in memory
        if cls.db_name != 'file::memory:?cache=shared':
            cls._conn = None
            _ = fn(cls, *args, **kwargs)
            cls._conn = make_connection(cls.db_name)
        else:
            rows = fn(cls, *args, **kwargs)
            if rows is not None and None not in rows:
                with contextlib.closing(make_connection(cls.db_name)) as conn:
                    with conn:
                        add_rows(conn, 'volumes', unnest(rows))
    return wrapped
