
#Compresses white space in strings to a single whitespace, up to 2^8 length whitespace
def compressWhiteSpace(s):
	#compresses and replaces nbsp chars with spaces
	s = s.strip().replace("\xc2\xa0", " ").replace("\xa0"," ").replace("\n"," ").replace("\t", " ")

	#this logarithmically reduces the number of whitespaces, 1/2 reduction per iteration
	for i in range(8):
		s = s.replace("  "," ")
	
	return s

"""
This is by far the best method for getting the natural language text of elements. 
Elem.itertext() doesn't work well because it includes the text within <script> tags, which is often
just more tags and metadata in textual form. This function does the same job as itertext, but
excludes script tags.

@elem: An lxml etree element object
"""
def getAllElementText(elem):
	text = ""

	for child in elem.iter():
	#for child in elem.findall(".//"):
		if child.tag != "script":
			if child.text is not None:
				text += (" " + child.text.strip())
			if child.tail is not None and len(child.tail.strip()) > 0:
				text += (" " + child.tail.strip())

	return text

"""
Returns first child beneath elem containing an attribute with attributeKey and attributeValue.

@softMatch: If true, softmatch attributeValue by only checking that the attributeKey value matches attributeValue for the length of attributeValue.
"""
def getChildByAttribute(elem, attributeKey, attributeValue, softMatch=False):
	for child in elem.findall(".//"):
		if attributeKey in child.attrib:
			if softMatch and child.attrib[attributeKey][0:len(attributeValue)] == attributeValue:
				return child
			elif child.attrib[attributeKey].strip() == attributeValue: #added strip(), since some attribute values appear to have trailing whitespace occasionally
				return child
	return None

#same as prior, but returns list of child hits
def getChildrenByAttribute(elem, attributeKey, attributeValue, softMatch=False):
	children = []
	for child in elem.findall(".//"):
		if attributeKey in child.attrib:
			if softMatch and child.attrib[attributeKey][0:len(attributeValue)] == attributeValue:
				children.append(child)
			elif child.attrib[attributeKey].strip() == attributeValue: #added strip(), since some attribute values appear to have trailing whitespace occasionally
				return children.append(child)
	return children

"""
Old versions of text stripping

def _getElementText(self,elem):
	ex = False
	try:
		text = etree.tostring(elem,method="text").strip() #etree.tostring() currently has a bug that causes it to throw here, for certain unicode chars
	except:
		ex = True
		#print("EXCEPT")
		#the dumb way
		text = ""
		if elem.text:
			text = elem.text.strip()
		if elem.tail:
			text += " " + elem.tail.strip()

	#if ex:
	#	print("E TEXT: "+str(text.encode()))
	return text

def _getAllElementText(self, elem):
	return " ".join(elem.itertext())
	text = ""
	#return elem.text_content()
	if elem.text is not None:
		text += elem.text
	if elem.tail is not None:
		text += elem.tail
	text += " ".join(elem.itertext(tag='a'))

	return text
"""

