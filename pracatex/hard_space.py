import re
import sys, glob

words = ['i', 'a', 'u', 'w', 'o', 'z']
for filename in glob.glob("*.tex"):
  print(filename)
  with open(filename, 'r') as f:
    text = f.read()
  for word in words:
    text = re.sub(' '+word+' ', ' '+word+'~', text)
    print('yes')
  with open(filename, 'w') as f:
    f.write(text)