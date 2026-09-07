from typing import TypedDict
# here typedict don't enforce the type of the values in the dictionary, but it helps us to know the types beforehand and also helps in code completion in IDEs like PyCharm, VSCode etc.
class Person(TypedDict):
    name: str
    age: int


new_person: Person = {
    "name": "Vinit",
    "age": 30
}

print(new_person)