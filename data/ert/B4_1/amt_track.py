from paraview.simple import * 
def start_cue(self):
	global annotations
	global maxIndex
	text_obj = Text()#make a text object
	annotations= []
	annotations.append('B4_1_NNF1')
	maxIndex = len(annotations)
def tick(self):
	global annotations
	global maxIndex
	index = int( self.GetClockTime() )
	if index >= maxIndex :
		 index = maxIndex - 1
	textSource = paraview.simple.FindSource('Text1')
	textSource.Text = annotations[index]
def end_cue(self): pass