def say_hello(name):
    print('hello {name}')

say_hello('tiffany')

class Puppy():
    def __init__(self, name, favorite_toy):
        self.name = name
        self.favourite_toy = favorite_toy

    def play(self):
        print(f"{self.name} is playing with the {self.favourite_toy}" )

marble = Puppy('Marble', 'teddy bear')

marble.play()

def add(a, b):
    return a + b

val = add(1, 2)

print(val)

def favourite_city(city):
    print(f"one of my favourite cities is {city}")

favourite_city('New York')
favourite_city('Paris')
