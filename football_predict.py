import numpy as np
import urllib.request
import json
import itertools

#Only for English Premier League
teamsIds = ['26','167','15','13','31','32','18','162','30','23','96','259','29','175','24',
'214','16','170','168','188']

#http://www.whoscored.com/Teams/32

#Data parse from http://www.whoscored.com
class ManageData:
	def __init__(*args, **kwargs):
		url = kwargs.get('url')
		path = kwargs.get('path')
		if url != None:
			self.data = self._readData(url)
		if path != None:
			self.data = self._loadData(path)

	def _readData(url):
		opener = urllib.request.build_opener()
		opener.addheaders = [('User-agent', 'Mozilla/5.0')]
		urllib.request.install_opener(opener)
		response = urllib.request.urlopen(url)
		return response.read().decode(response.headers.get_content_charset())


	#load data from json file file
	def _loadData(path):
		data = None
		try:
			f = open(path)
			data = f.read()
		except Exception as e:
			raise "File not found"
		return json.loads(data)

	#http://www.whoscored.com/Teams/32/Show/-Manchester-United
	def getPlayersFromTeam(team):
		''' Get list of players from team
		'''
		data = readData(team)
		s = 'defaultTeamPlayerStatsConfigParams.defaultParams'
		start = data.find(s)
		end = data.find('}]);', start)
		players = data[start+297:end+2]
		result = json.loads(players)
		mostDribled = 0
		name = ''
		return result[0]['TeamName'], result
	

	def getTeamData(ids):
		url = 'http://www.whoscored.com/Teams/'
		values ={}
		for iddata in ids:
			team, data = self.getPlayersFromTeam(url + iddata)
			values[team] = data
		with open('teams','w') as outfile:
			json.dump({'teams': values}, outfile)



def getStat(result):
	for r in result:
		if int(r['TotalPasses']) != 0:
			dr = int(r['AccuratePasses']) / int(r['TotalPasses'])
			if dr > mostDribled:
				mostDribled = dr
				name = r['LastName']

def parseOnlineTextMatch(url):
	'''
		http://www.whoscored.com/Matches/829535/Live
	'''
	pass

def getActivity():
	data = readData('http://www.whoscored.com/Players/3859')
	startstat = data.find('defaultWsPlayerStatsConfigParams.defaultParams')+49
	endstat = data.find('var', startstat)-22
	result = json.loads(data[startstat:endstat])
	result[0]['KnownName'] = 'Wayne Rooney'

def getPlayersFromTeamByPos(teamsdata, team, pos):
	currteam = teamsdata[team]
	for curr in currteam:
		if curr['PositionShort'] ==pos:
			yield curr

def getPlayer(teamdata, lastname):
	pass
	
def comparePlayers(teamsdata, player, teams):
	'''
		Compare player with players from other team on same position
	'''
	return teams


def getBest(teamsdata, crireria,limit=None):
	values = []
	for team in teams.keys():
		for t in teams[team]:
			values.append((t[crireria], t['LastName'], t['TeamName']))
	if limit == None:
		limit = len(values)
	return list(reversed(sorted(values)))[0:limit]

def getPos(sorttuple, player):
	'''
		Input is sorted tuple [(Player, value)]
		Case, when not target player in list
	'''
	return len(list(itertools.takewhile(lambda x: x[1] != player, sorttuple)))


class OptimalTeam:
	def __init__(self, teamdata, team, opteam):
		self.teamdata = teamdata
		self.team = team
		self.opteam = opteam

	def choose(self):
		'''
			Выбор защиты основывается на уровне нападения
		'''
		pass


def getOptimalTeam(teamsdata, team, formation, opteam):
	'''
		formation can be 4-4-2 or 3-5-2, but not 4-2-1-3 
		Если более сильная атака, то выбираем мощную защиту и наоборот
		opteam-opposite team
	'''
	if team not in teamsdata:
		raise "This team not in base"
	data = teamsdata[team]
	players = list(getPlayersFromTeamByPos(teamsdata, team, 'GK'))

def GkToForward(player, gk):
	''' Соотношение удара по воротам и отбитым мячам'''
	if gk[0] == 0:
		return 0
	return player[0]/gk[0]
