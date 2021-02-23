from zipfile import ZipFile

def writeScores(method_name,scores):
    fn = "predictions.txt"
    print("")
    with open(fn, 'w') as output_file:
        for idx,x in enumerate(scores):
            #out =  metrics[idx]+":"+str("{0:.2f}".format(x))+"\n"
            #print(out)
            output_file.write(f"{x}\n")
            
   
    with ZipFile("en-de.zip","w") as newzip:
    	newzip.write("predictions.txt")
     
