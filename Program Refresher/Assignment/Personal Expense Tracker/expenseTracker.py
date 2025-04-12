from datetime import date as dt

# class to track expenses
class ExpenseTracker:
    def __init__(self):
        self.expenses = []
    # storing expence as a list of dictionaries
    def add_expense(self, date, category, amount, description):
        expense = {
            'date': date,
            'category': category,
            'amount': amount,
            'description': description
        }
        self.expenses.append(expense)

    def get_expenses(self,month=None, year=None):
        if month and year:
            #added defaults to make data entry easier
            return [expense for expense in self.expenses if expense['date'].startswith(f"{year}-{month:02d}")]
        elif month:
            return [expense for expense in self.expenses if expense['date'].startswith(f"{month:02d}")]
        elif year:
            return [expense for expense in self.expenses if expense['date'].startswith(str(year))]
        else:
            return self.expenses
    #save to file
    def save_expenses(self, filename):
        with open(filename, 'w') as file:
            for expense in self.expenses:
                file.write(f"{expense['date']},{expense['category']},{expense['amount']},{expense['description']}\n")
    #read from file
    def load_expenses(self, filename):
        try:
            with open(filename, 'r') as file:
                for line in file:
                    try:
                        date, category, amount, description = line.strip().split(',')                    
                        self.add_expense(date, category, float(amount), description)
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty expense list.")
    
#input from user
def input_expense(tracker):
    date = input("Enter the date (YYYY-MM-DD): ") or dt.today().strftime("%Y-%m-%d")
    category = input("Enter the category: ") or "Miscellaneous"
    amount = float(input("Enter the amount: ") or "0.0")
    description = input("Enter a description: ") or "No description"
    tracker.add_expense(date, category, amount, description)

def display_expenses(tracker):
    expenses = tracker.get_expenses()
    if not expenses:
        print("No expenses recorded.")
        return

    print("\nExpenses:")
    for expense in expenses:
        print(f"Date: {expense['date']}, Category: {expense['category']}, Amount: {expense['amount']}, Description: {expense['description']}")

def main():
    tracker = ExpenseTracker()
    tracker.load_expenses('expenses.csv')

    #interactive menu
    while True:
        print("\nExpense Tracker")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Save Expenses")
        print("4. Track budget")
        print("5. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            input_expense(tracker)
        elif choice == '2':
            display_expenses(tracker)
        elif choice == '3':
            tracker.save_expenses('expenses.csv')
            print("Expenses saved.")
        elif choice == '4':
            budget = float(input("Enter your budget: "))
            total_expenses = sum(expense['amount'] for expense in tracker.get_expenses(month=dt.today().month, year=dt.today().year))
            print(f"Total expenses for the month: {total_expenses:.2f}")
            if total_expenses > budget:
                print(f"Warning: You have exceeded your budget by {total_expenses - budget:.2f}.")
            else:
                print(f"You are within your budget. You have {budget - total_expenses:.2f} remaining.")
        elif choice == '5':            
            exit_choice = input ("Press 1 to save and exit, any other key to just exit...")
            if exit_choice == '1':
                tracker.save_expenses('expenses.csv')
                print("Expenses saved.")
                break
            else:
                print("Exiting without saving.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()