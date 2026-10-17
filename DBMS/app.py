import sys 
from dbhelper import DB_Helper

class Flipkart_Login_System:

    def __init__(self):
        # Connecting to database 
        self.database = DB_Helper()
        self.menu()


    def menu(self):

        user_input = input("""
            1. Enter 1 to register
            2. Enter 2 to login
            3. Anything else to leave
        """)

        if user_input == "1":
            self.register()
        elif user_input == "2":
            self.login()
        else:
            sys.exit()

    def register(self):
        name = input("Enter your name: ")
        email = input("Enter your email: ")
        password = input("Enter your password: ")

        response = self.database.register_into_database(name, email, password)
        if(response == 1):
            print("Registration Successful")
        else:
            print("Registration Failed")

        self.menu()

    def login(self):

        email = input("Enter Email: ")
        password = input("Enter Password: ")

        data = self.database.search_user(email, password)
        if(len(data) == 0):
            print("Incorrect Email/Password")
            self.logged_in_menu()
        else:
            print("Hello", data[0][1])

    def logged_in_menu(self):
        option = input("""
            1. Enter 1 to see profile.
        """)
            
object = Flipkart_Login_System()