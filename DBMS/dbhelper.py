import mysql.connector
import sys

# Hierarchy
# Database Server <- Tables <- Rows and Cols <- User Data
# We will create a data base for login and registration data

# In almost all project, we only do 4 operations with database
# Create, Retrieve, Update, Delete

class DB_Helper:

    def __init__(self):

        try:
            self.conenction = mysql.connector.connect(
                host="localhost", user='root', password="", database='basic_project'
            )

            self.mycursor = self.conenction.cursor()
        except:
            print("Some Error Occured, Could not connect to database")
            sys.exit(0)
        else:
            print("Connected to Database")

    def register_into_database(self, name, email, password):
        try:

            # Executing Query Directly
            self.mycursor.execute("INSERT INTO users (user_id, name, email, password) VALUES (NULL, %s, %s, %s)", (name, email, password))

            self.conenction.commit()

        except:
            return -1
        else:
            return 1

    def search_user(self, email, password):

        self.mycursor.execute("SELECT * FROM users WHERE email LIKE %s AND password LIKE %s", (email, password))
        data = self.mycursor.fetchall()

        return data