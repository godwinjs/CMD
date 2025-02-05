class Dog :
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def bark(self):
        print('bark')
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    
class Attendee:
    'Common base class for all attendees'

    def __init__(self, name, tickets):
        self.name = name
        self.tickets = tickets

    def displayAttendee(self):
        print('Name : {}, Tickets: {}'.format(self.name, self.tickets) )
    
    def addTickets(self):
        self.tickets += 1
        print(f'{self.name} tickets increased {self.tickets}' )

attendee1 = Attendee('B John', 2)
attendee1.displayAttendee()

attendee2 = Attendee('A Otega', 1)
attendee2.addTickets()

### End of ch2.py ###

## Advanced ##
# Memory Management
# Python uses a system called reference counting to manage memory.
# When an object is created, Python allocates memory for the object and
# assigns a reference count of 1 to that object.
# As the object is passed around in the program, Python increments and decrements
# the reference count as necessary.
# When the reference count reaches 0, Python deallocates the memory for the object.

# Garbage Collection: Python also uses a system called garbage collection to manage memory. (Java, C#, ruby) support garbage collection.
# Garbage collection is a process that automatically reclaims memory that is no longer in use.

# Intro to multithreading
# Multithreading is the ability of a CPU to execute multiple processes or threads concurrently.

# Multithreading in Python
# Python has a module called threading that provides a simple way to create and manage threads.

# Intro to algorithm
# An algorithm is a set of instructions that are used to solve a specific problem or perform a specific task.