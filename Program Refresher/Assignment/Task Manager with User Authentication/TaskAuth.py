class hash_password:
    def __init__(self, password, hashed = False):
        if hashed == True:
            self.password = password
        else:
            if self.validate_password(password) == False:
                raise ValueError("Password must be at least 8 characters long and contain both letters and numbers.")
            import hashlib
            self.password =  hashlib.sha256(password.encode()).hexdigest()   
    def __str__(self):
        return self.password
    def __eq__(self, hashed_password:str):
        return self.password == hashed_password
    def validate_password(self, password:str):
        if len(password) < 8:
            return False
        if not any(char.isdigit() for char in password):
            return False
        if not any(char.isalpha() for char in password):
            return False
        return True
    
class authuser:
    def __init__(self, username, password, hashed = False):
        self.username = username
        self.password = hash_password(password, hashed)

    def __eq__(self, other):
        if isinstance(other, authuser):
            return self.username == other.username and self.password == other.password
        return False

    def __str__(self):
        return f"{self.username},{self.password}"

class task:
    def __init__(self, task_description, username, task_id_gen):
        self.username = username
        self.task_description = task_description
        self.status = False  # False for Pending, True for complete
        self.task_id = next(task_id_gen)  # Generate a unique task ID using the generator

    def __str__(self):
        #return f"Task: {self.task_description}, Status: {'Complete' if self.status else 'Pending'}, Assigned to: {self.username}"   
        return f"{self.task_description},{'Complete' if self.status else 'Pending'},{self.username}"

    def __eq__(self, other):
        if isinstance(other, task):
            return self.task_id == other.task_id and self.username == other.username and self.task_description == other.task_description
        elif isinstance(other, int):
            return self.task_id == other        
        return False
    
    def mark_complete(self):
        self.status = True
    def mark_pending(self):
        self.status = False
    def get_task(self):
        return {"description":self.task_description, "status": self.status, "username": self.username, "id": self.task_id}

def read_tasks_from_file(username, task_id_gen):
    tasks = []
    filename = f"{username}_tasks.txt"
    try:
        with open(filename, 'r') as file:
            for line in file:
                task_description, status, username = line.strip().split(',')
                task_obj = task(task_description, username, task_id_gen)
                if status == 'Complete':
                    task_obj.mark_complete()
                else:
                    task_obj.mark_pending()
                tasks.append(task_obj)
    except FileNotFoundError:
        print(f"File {filename} not found. Starting with an empty task list.")
    return tasks

def write_tasks_to_file(tasks, username):
    filename = f"{username}_tasks.txt"
    with open(filename, 'w') as file:
        for task_obj in tasks:
            file.write(f"{task_obj}\n")
#to generate task IDs
def id_generator(start=0, step=1):
    current_id = start
    while True:
        yield current_id
        current_id += step
    
def task_menu(username):
    # Initialize the task ID generator
    task_id_gen = id_generator(0,1)
    tasks = read_tasks_from_file(username,task_id_gen)    

    while True:
        print(f"Task Menu for {username}:")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task Complete")
        print("4. Delete Task")
        print("5. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            task_description = input("Enter task description: ")
            new_task = task(task_description, username, task_id_gen)
            tasks.append(new_task)
            print(f"Task '{task_description}' added successfully.")
        elif choice == '2': 
            print("Tasks:")
            for task_obj in tasks:
                id = task_obj.get_task()["id"]
                status = "Complete" if task_obj.status else "Pending"                
                print(f"ID: {id}, Description: {task_obj.task_description}, Status: {status}, Assigned to: {task_obj.username}")    
                print(task_obj)
        elif choice == '3':     
            task_id = input("Enter task id to mark as complete: ")
            for task_obj in tasks:
                import pdb; pdb.set_trace()
                if task_obj == int(task_id):
                    task_obj.mark_complete()
                    print(f"Task '{task_obj.task_description}' marked as complete.")
                    break
            else:
                print("Task not found.")
        elif choice == '4':
            task_id = input("Enter task id to delete: ")
            for task_obj in tasks:
                if task_obj == int(task_id):
                    task_description = task_obj.task_description
                    tasks.remove(task_obj)
                    print(f"Task '{task_description}' deleted successfully.")
                    break
            else:
                print("Task not found.")
        elif choice == '5':
            write_tasks_to_file(tasks, username)
            print("Exiting task menu.")
            break
def write_users_to_file(users):
    with open('users.txt', 'w') as file:
        for user in users:
            file.write(f"{user}\n")

def read_users_from_file():
    users = []
    try:
        with open('users.txt', 'r') as file:
            for line in file:
                username, password = line.strip().split(',')
                user_obj = authuser(username, password, hashed=True)
                users.append(user_obj)
    except FileNotFoundError:
        print("No existing users found.")
    return users

def auth_menu():
    users = read_users_from_file()
    try:
        while True:
            print("Authentication Menu")
            print("Please choose an option:")
            print("1. Register")
            print("2. Login")
            print("3. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                username = input("Enter username: ")
                password = input("Enter password: ")
                try:
                    # Check if username already exists
                    if any(user.username == username for user in users):
                        print("Username already exists. Please choose a different one.")
                        continue
                    # Create a new user and add to the list
                    new_user = authuser(username, password)
                    users.append(new_user)
                    print(f"User {username} registered successfully.")
                except ValueError as e:
                    print(e)

            elif choice == '2':
                username = input("Enter username: ")
                password = input("Enter password: ")
                for user in users:
                    if user == authuser(username, password):
                        print(f"Welcome {username}!")
                        task_menu(username)
                        break
                else:
                    print("Invalid username or password.")

            elif choice == '3':
                break

            else:
                print("Invalid choice. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        write_users_to_file(users)
        print("User data saved. Exiting authentication menu.")


def main():
    auth_menu()
if __name__ == '__main__':
    main()
