import numpy as np
import urllib.request
import json
import itertools
from collections import Counter, namedtuple
import itertools
from fn import F, op, _
from fn import recur
from fn.iters import take, drop, map, filter

#Only for English Premier League
teamsIds = ['26','167','15','13','31','32','18','162','30','23','96','259','29','175','24',
'214','16','170','168','188']

#http://www.whoscored.com/Teams/32

#Data parse from http://www.whoscored.com
class ManageData:
	def __init__(self, *args, **kwargs):
		url = kwargs.get('url')
		path = kwargs.get('path')
		if url != None:
			self.data = self._readData(url)
		if path != None:
			self.data = self._loadData(path)
		self.teams = self.data['teams']

	def _readData(self, url):
		opener = urllib.request.build_opener()
		opener.addheaders = [('User-agent', 'Mozilla/5.0')]
		urllib.request.install_opener(opener)
		response = urllib.request.urlopen(url)
		return response.read().decode(response.headers.get_content_charset())


	#load data from json file file
	def _loadData(self, path):
		data = None
		try:
			f = open(path)
			data = f.read()
		except Exception as e:
			raise "File not found"
		return json.loads(data)

	#http://www.whoscored.com/Teams/32/Show/-Manchester-United
	def getPlayersFromTeam(self, team):
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
	

	def getTeamData(self, ids):
		url = 'http://www.whoscored.com/Teams/'
		values ={}
		for iddata in ids:
			team, data = self.getPlayersFromTeam(url + iddata)
			values[team] = data
		with open('teams','w') as outfile:
			json.dump({'teams': values}, outfile)

	def getBest(self, param, limit=None):
		'''
			Return list of best playerts by some param
			For example, get "best" players by Fouls
		'''

		@recur.tco
		def recurgetBest(teamsdata, teams, param, data=[]):
			if len(teams) == 0:return False, data
			team = teams.pop()
			newvalue = [(t[param], t['LastName'], t['TeamName']) for t in teamsdata[team]]
			return True, (teamsdata, teams, param, data+newvalue, )

		values = recurgetBest(self.teams, list(self.teams.keys()), param)
		if limit == None:
			limit = len(values)
		return list(reversed(sorted(values)))[0:limit]

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
	print({'Rooney' : result[0]})
	#parseData(data)
	'''for d in data.split('\n'):
		print(d.find('defaultWsPlayerStatsConfigParams.defaultParams'))'''

def getPlayersFromTeamByPos(teamsdata, team, pos):
	return list(filter(lambda x: x['PositionShort'] == pos, teamsdata[team]))

def getPlayer(teamdata, lastname):
	pass
	
def comparePlayers(teamsdata, player, teams):
	'''
		Compare player with players from other team on same position
	'''
	return teams

def getPos(sorttuple, player):
	'''
		Input is sorted tuple [(Player, value)]
		Case, when not target player in list
	'''
	return len(list(itertools.takewhile(lambda x: x[1] != player, sorttuple)))


class OptimalTeamException(Exception):
	pass

class OptimalTeam:
	def __init__(self, teamdata):
		self.teamdata = teamdata

	def choose(self):
		'''
			Выбор защиты основывается на уровне нападения
		'''
		pass

	def getLocalResults(self, players, vec):
		''' 
			Получить результат, основываясь только на данных одноклубников
			vec - Вектор параметров
		'''
		print(list(map(lambda x: (x['TotalClearances'], x['Rating'], x['GameStarted'], \
		x['ManOfTheMatch'], x['AerialWon'], ), players)))

	def getOptimalTeam(self, team, opteam, formation):
		'''
			formation can be 4-4-2 or 3-5-2, but not 4-2-1-3 
			Если более сильная атака, то выбираем мощную защиту и наоборот
			opteam-opposite team
			GK - Goalkeeper
			D(CR) - Defence Right
			D(LR) - Defence Left
			D(L) - Defence Left
			FW - Forward
			AM - Attack mid
		'''
		if team not in self.teamdata:
			raise  "This team not in base"
		position = ['DF', 'FW']
		form_data = {'GK':1}
		form_res = list(map(lambda x: int(x), formation.split('-')))
		GK = self._chooseGK(opteam)
		self._chooseDefence(team, opteam, form_res[0])

	def _getParamValues(self, players, values):
		return list(map(lambda x: [x[p] for p in values], players))

	def _chooseGK(self, team):
		pos = 'GK'
		players = list(getPlayersFromTeamByPos(self.teamdata, team, pos))
		params = ['TotalClearances', 'Rating', 'GameStarted', 'ManOfTheMatch', 'AerialWon']
		vecparams = self._getParamValues(players, params)
		matr = np.array(vecparams)
		c = Counter(np.argmax(matr, axis=0)).most_common(1).pop()
		return players[c[0]]['LastName']

	def _chooseDefence(self, team, opteam, num):
		'''
			Брать во внимание уровень нападающих в команде соперников
		'''
		positions = self._getDefences(num)
		center = self._chooseDefenceCenter(team, opteam)
		'''for pos in positions:
			players = list(getPlayersFromTeamByPos(self.teamdata, team, pos))
			vecparams = list(map(lambda x: [x['AerialWon']], players))
			break'''

	def _chooseDefenceCenter(self, team, opteam):
		players = list(getPlayersFromTeamByPos(self.teamdata, team, 'D(C)'))
		params = ['TotalTackles', 'AerialWon', 'Rating','OffsidesWon','GameStarted',\
		 'ShotsBlocked', 'LastName']
		tomaxvalues = self._getParamValues(players, params)
		tominvalues = self._getParamValues(players,['AerialLost','Dispossesed','Yellow'])
		maxv = self._optimalPlayers(np.array(tomaxvalues), np.argmax, 2, players)
		minv = self._optimalPlayers(np.array(tominvalues), np.argmin, 2, players)
		#if same players both in maxv and minv append in result list
		data = list(maxv.intersection(minv))
		self._opteamOptimal({'Dribbles': ['WasDribbled'], \
			'TotalShots': ['ShotsBlocked'], 'Goals':['ShotsBlocked']}, \
			opteam, players)
		if len(data) == 2:
			return data



	def _optimalPlayers(self, matr, func, num, players):
		c = Counter(func(matr, axis=0)).most_common(num)
		return set(map(lambda x: players[x[0]]['LastName'], c))

	def _opteamOptimal(self, params, opteam, teamdata):
		'''
			Best players from opposite team
		'''
		gs = 'GameStarted'
		opplayers = list(filter(lambda x:x[gs] > 0, \
			getPlayersFromTeamByPos(self.teamdata, opteam, 'FW')))
		#values = list(filter(lambda x: x[-1:][0] > 0, self._getParamValues(opplayers, params)))
		for v in params.keys():
			print(opplayers)
			for player in teamdata:
				#preresult5 = list(F() << (_/player[gs]) << (filter, _ > 0))
				preresult = list(
					map(_/player[gs],
					filter(_ > 0, \
					map(lambda x: player[x], params[v]))
					))
				print(preresult)

	def _getDefences(self, num):
		if num <= 2:
			raise OptimalTeamException("Number of defences can't be less than 2")
		pos = ['D(L)', 'D(R)'] + list(itertools.repeat('D(C)', num-2))
		return pos



class Statistics:
	'''
		Statistics and correlations for parameters in data
	''' 
	def __init__(self, teamsdata):
		self.teamdata = teamsdata

	def compare(self, first, second):
		'''
			Compare some two parameters
			st = Statistics(teams)
			st.compare('Height', 'AerialWon')
		'''
		bans = self.teamdata
		keys = list(self.teamdata.keys())
		result = []
		for targteam in keys:
			result.append(list(map(lambda x: [x[first], x[second],x['GameStarted']], bans[targteam])))
		return sorted(result, key=lambda x:x[2])


	def predict(self, params, predvalue):
		'''
			params - data for prediction
			predvalue - prediction value
		'''
		pass

def GkToForward(player, gk):
	''' Соотношение удара по воротам и отбитым мячам'''
	if gk[0] == 0:
		return 0
	return player[0]/gk[0]
