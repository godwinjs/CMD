miles = input('Enter distance in miles')
kilometers_value = float(miles) * 1.609344

print(f'{miles} Miles in Kilometers is: {kilometers_value}')

infile = open('values.txt', 'rt')
outfile = open('values_totaled.txt', 'wt')
print('Processing input')

sum = 0
for line in infile:
    sum += int(line)
    print(line.rstrip(), file=outfile)
print('\nTotal: ' + str(sum), file=outfile )
outfile.close()
print('Output complete')

def checkTemp(temp):
    if temp < 15:
        return 'wear a jacket'
    elif temp >= 25 and temp <= 35:
        return 'pack a jacket'
    else:
        return 'Leave the jacket at home'
    
print(checkTemp(10))
print(checkTemp(30))
print(checkTemp(37))