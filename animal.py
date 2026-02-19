# Base Animal Class
class Animal:
    def __init__(self, name, age, species):
        self.name = name
        self.age = age
        self.species = species
        self.vaccinations = []  # List to store vaccinations
    
    def add_vaccination(self, vaccine_name):
        """Add a vaccination to the animal's vaccination list"""
        self.vaccinations.append(vaccine_name)
    
    def display_info(self):
        """Display animal information"""
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Species: {self.species}")
        print(f"Vaccinations: {', '.join(self.vaccinations) if self.vaccinations else 'None'}")


# Subclass: Dog inherits from Animal
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age, "Dog")
        self.breed = breed
    
    def display_info(self):
        """Display dog information including breed"""
        super().display_info()
        print(f"Breed: {self.breed}")
    
    def bark(self):
        """Dog bark method"""
        print(f"{self.name} says: Woof! Woof!")


# Subclass: Cat inherits from Animal
class Cat(Animal):
    def __init__(self, name, age, color):
        super().__init__(name, age, "Cat")
        self.color = color
    
    def display_info(self):
        """Display cat information including color"""
        super().display_info()
        print(f"Color: {self.color}")
    
    def meow(self):
        """Cat meow method"""
        print(f"{self.name} says: Meow!")


# Example usage
if __name__ == "__main__":
    # Create an animal instance
    animal = Animal("Buddy", 3, "Generic Animal")
    animal.add_vaccination("Rabies")
    animal.add_vaccination("Distemper")
    print("=== Animal Info ===")
    animal.display_info()
    print()
    
    # Create a Dog instance
    dog = Dog("Max", 5, "Golden Retriever")
    dog.add_vaccination("Rabies")
    dog.add_vaccination("Parvovirus")
    dog.add_vaccination("Bordetella")
    print("=== Dog Info ===")
    dog.display_info()
    dog.bark()
    print()
    
    # Create a Cat instance
    cat = Cat("Whiskers", 2, "Orange")
    cat.add_vaccination("Rabies")
    cat.add_vaccination("Feline Leukemia")
    print("=== Cat Info ===")
    cat.display_info()
    cat.meow()
