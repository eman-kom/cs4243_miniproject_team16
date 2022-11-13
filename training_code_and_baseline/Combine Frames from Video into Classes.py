import os
inFolder = ""
outFolder = ""

for root, dirs, files in os.walk(inFolder):
	for name in files:
		fname = os.path.join(root, name)
		vid = f.split('\\')[6]
		className = f.split('\\')[5]
		newPath = os.path.join(dd, className+"\\"+vid+"_"+name)
		os.rename(fname, newPath)
