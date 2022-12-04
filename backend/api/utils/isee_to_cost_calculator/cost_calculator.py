#This function takes a float in input (isee) and returns a float which represents how much the student has to pay
def cost_calculator(isee):
    if isee == 0: #isee not provided to the system
        return "7.10"
    if isee <= 19469:
        return "2.20"
    elif isee >= 19469.01 and isee <= 35679:
        return "3.00"
    elif isee >= 35679.01 and isee <= 60209:
        return "4.10"
    else:
        return "5.90"