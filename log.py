from datetime import datetime
 
def log(text):
    print(text)
    with open('out.log', 'a') as f:
        f.write("\n"+str(datetime.now()) + " -> " + text)