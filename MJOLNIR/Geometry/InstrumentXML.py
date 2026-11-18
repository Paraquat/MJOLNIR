#!/usr/bin/env python3
"""
Created on Mon Mar 12 16:43:47 2018

@author: lass
"""
import numpy as np
def parseXML(filename):
		
	from MJOLNIR.Geometry import Detector,Analyser,Wedge,Instrument
	import xml.etree.ElementTree as ET
	import numpy as np

	tree = ET.parse(filename)
	instr_root = tree.getroot()

	instrSettings = {}
		
	for attrib in instr_root.keys():
		instrSettings[attrib]=instr_root.attrib[attrib]

	Instr = Instrument.Instrument(**instrSettings)

	for wedge in list(instr_root):#.getchildren():
		
		if wedge.tag in dir(Wedge):
			Wedgeclass_ = getattr(Wedge, wedge.tag)
		else:
			raise ValueError(f"Element is supposed to be a Wedge, but got '{wedge.tag}'.")
		wedgeSettings = {}
		
		for attrib in wedge.keys():
			if attrib=='concept':
				wedgeSettings[attrib]=np.array(wedge.attrib[attrib].strip().split(','),dtype=str)
			else:		
				wedgeSettings[attrib]=np.array(wedge.attrib[attrib].strip().split(','),dtype=float)
			
		temp_wedge = Wedgeclass_(**wedgeSettings)
		#print(temp_wedge)
		
		
		
		for item in list(wedge):#.getchildren():
			if item.tag in dir(Detector):
				class_ = getattr(Detector, item.tag)
			elif item.tag in dir(Analyser):
				class_ = getattr(Analyser,item.tag)
			else:
				raise ValueError(f"Item '{item.tag}' not recognized as MJOLNIR detector or analyser.")
			
			itemSettings = {}
			for attrib in item.keys():
				attribVal = item.get(attrib).strip().split(',')
				if len(attribVal)==1:
					itemSettings[attrib]=float(attribVal[0])
				else:
					if(attrib=='split'):
						#print(type(attribVal))
						itemSettings[attrib]=attribVal
					else:
						itemSettings[attrib]=np.array(attribVal,dtype=float)	
			try:
				temp_item = class_(**itemSettings)
			except TypeError as e:
				print(e.args[0])
				raise ValueError('Item {} misses argument(s):{}'.format(class_,e.args[0].split(':')[1]))
			except ValueError:
				raise ValueError(f'Item {class_} not initialized due to error.')
			#print(temp_item)
			temp_wedge.append(temp_item)
			#print()

		#print(str(temp_wedge))
		Instr.append(temp_wedge)
	return Instr

def createXMLString(instrument):
	XMLString = '<?xml version="1.0"?>\n'
	XMLString+= '<Instrument '
	for attrib in instrument.settings:
		XMLString+=f"{attrib}='{instrument.settings[attrib]}' "
	XMLString+='>\n'
		
	for wedge in instrument.wedges:
		XMLString+="\t<Wedge "
		for attrib in wedge.settings:
			XMLString+=f"{attrib}='{wedge.settings[attrib]}' "
		XMLString+='>\n'

		for item in wedge.analysers + wedge.detectors:
			itemClass = str(item.__class__).split('.')[-1][:-2]
			XMLString+=f"\t\t<{itemClass}"
			for key in item.__dict__:
				value = item.__getattribute__(key)
				if isinstance(value,type(np.array([0,0,0]))):
					valueStr = ','.join([str(x) for x in item.__getattribute__(key)])
				else:
					valueStr = str(value)
				XMLString+=f" {str(key)[1:]}='{valueStr}'"
			XMLString+=f"></{itemClass}>\n"
			
		
		XMLString+="\t</Wedge>\n"
	XMLString+="</Instrument>\n"
	return XMLString
		
