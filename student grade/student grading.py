# Simple AI-style grader for multiple subjects

def grade_from_marks(marks):
    """Return a letter grade for the given marks."""
    if marks >= 90:
        return "A+"
    elif marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 50:
        return "D"
    else:
        return "F"

def main():
    subjects = {}
    n = int(input("How many subjects? "))
    
    for i in range(n):
        name = input(f"\nEnter name of subject {i+1}: ")
        while True:
            try:
                marks = float(input(f"Enter marks for {name} (0–100): "))
                if 0 <= marks <= 100:
                    break
                else:
                    print("Marks must be between 0 and 100.")
            except ValueError:
                print("Please enter a number.")
        subjects[name] = marks

    print("\n--- Report Card ---")
    total = 0
    for subject, marks in subjects.items():
        grade = grade_from_marks(marks)
        total += marks
        print(f"{subject:15}: {marks:5.1f} -> Grade {grade}")

    average = total / n
    print(f"\nOverall Average: {average:.1f}")
    print(f"Overall Grade : {grade_from_marks(average)}")

if __name__ == "__main__":
    main()
