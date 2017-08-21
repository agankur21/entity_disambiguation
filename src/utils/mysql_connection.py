from decorator_utils import *
import mysql.connector
from config.mysql import *


class MySqConnector:
    @raise_exception
    def __init__(self, user=USERNAME, host=HOST, database=DATABASE):
        print ('Initialising connection to database')
        self.connection = mysql.connector.connect(user=user, host=host, database=database)

    def __del__(self):
        print("Closing connection to database")
        self.connection.close()

    @raise_exception
    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        result = []
        for out in cursor:
            result.append(out)
        cursor.close()
        return result


if __name__ == '__main__':
    conn = MySqConnector()
    result = conn.execute_query('select STR from umls.mrconso where str like "%skin tumour%"')
    print result
