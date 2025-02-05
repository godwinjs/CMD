def say_hello(name):
    print(f'hello {name}')

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

spices = ['salt', 'pepper', 'cumin', 'turmeric', 'paprika']

for spice in spices:
    print(spice)

i = 5

while i <= 100:
    print(i)
    i += 5

fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']

print('our fruits selection:')

for fruit in fruits:
    print(fruit)

print('end friuts selection')

import imp

# name = input('what is your name: ')

# imp.greet(name)

x = int('50')
y = float('4.55')

print(x+y)

first_name = 'giddy'
last_name = 'hon'
note = 'award: nobel peace prize'

first_name_cap = first_name.capitalize()

print(first_name_cap +" "+ last_name)

award_location = note.find('award: ')

print(award_location)

award_text = note[7:]
print(award_text)


import re

five_digit_zip = '98101'
nine_digit_zip = '98101-0003'
phone_number = '234-567-8901'

#r let the compiler know that there might be a backslash in the string that should be ignored
#\d search for digits
#{5} how many digits to find
five_digit_exp = r'\d{5}'
print(re.search(five_digit_exp, five_digit_zip))
print(re.search(five_digit_exp, nine_digit_zip))
print(re.search(five_digit_exp, phone_number))