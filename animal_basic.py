# Simple Animal Class for Beginners

class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.vaccinations = []  # Empty list for vaccinations
    
    def add_vaccine(self, vaccine):
        self.vaccinations.append(vaccine)

# Dog is a subclass of Animal
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed

# Create a dog instance
my_dog = Dog("Max", "Golden Retriever")

# Add vaccinations to the list
my_dog.add_vaccine("Rabies")
my_dog.add_vaccine("Distemper")

# Print results
print(f"Dog Name: {my_dog.name}")
print(f"Species: {my_dog.species}")
print(f"Breed: {my_dog.breed}")
print(f"Vaccinations: {my_dog.vaccinations}")
