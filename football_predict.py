import numpy as np
import urllib.request
import json
import itertools
from collections import Counter, namedtuple
import itertools
from fn import F, op, _
from fn import recur
from fn.iters import take, drop, map, filter
import textblob

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
	return list(filter(lambda x: x['LastName'] == lastname, teamdata))[0]


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
		self.fwparams = {'Dribbles': ['WasDribbled'], \
		'TotalShots': ['ShotsBlocked'], 'Goals':['ShotsBlocked']}

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
		result = {}
		if team not in self.teamdata:
			raise OptimalTeamException("This team not in base")
		form_res = list(map(lambda x: int(x), formation.split('-')))
		if len(form_res) != 3:
			raise OptimalTeamException("Error in formation representation")
		result['GK'] = [self._chooseGK(team)]
		result['DF'] = self._chooseDefence(team, opteam, form_res[0])
		result['MF'] = self._chooseMidfielder(team, form_res[1])
		result['FW'] = self._chooseForward(team, opteam, form_res[2])
		return result

	def _getParamValues(self, players, values):
		return list(map(lambda x: [x[p] for p in values], players))

	def _getTargetPlayers(self, team, num, pos, params):
		pos = pos
		players = list(getPlayersFromTeamByPos(self.teamdata, team, pos))
		params = ['TotalClearances', 'Rating', 'GameStarted', 'ManOfTheMatch', 'AerialWon']
		vecparams = self._getParamValues(players, params)
		matr = np.array(vecparams)
		c = Counter(np.argmax(matr, axis=0)).most_common(num).pop()
		return players[c[0]]['LastName']

	def _chooseGK(self, team):
		pos = 'GK'
		players = list(getPlayersFromTeamByPos(self.teamdata, team, pos))
		params = ['TotalClearances', 'Rating', 'GameStarted', 'ManOfTheMatch', 'AerialWon']
		return self._getTargetPlayers(team, 1, pos, params)

	def _chooseMidfielder(self, team, num):
		if num == 0:
			raise OptimalTeamException("Count of midfielders is zero")

		def getMFCenter(func, team, num):
			pos = 'M(C)'
			params = ['Rating', 'TotalPasses', 'KeyPasses', 'GameStarted']
			return func(team, num, pos, params)

		def getMFLR(self, team, num):
			pass
		center = getMFCenter(self._getTargetPlayers, team,2)
		return []

	def _chooseDefence(self, team, opteam, num):
		'''
			Брать во внимание уровень нападающих в команде соперников
		'''
		result = []
		positions = self._getDefences(num)
		result += self._chooseDefenceCenter(team, opteam, Counter(positions)['D(C)'])
		result += self._chooseDefenceLR(team, num)
		return result

	def _chooseDefenceLR(self, team, num):
		result = []
		if num == 1:
			pass

		def get(idv):
			params = ['KeyPasses', 'Dribbles', 'TotalPasses', 'Rating', 'OffsidesWon', \
			'GameStarted']
			playersL = list(getPlayersFromTeamByPos(self.teamdata, team, idv))
			tomaxvalues = self._getParamValues(playersL, params)
			maxv = self._optimalPlayers(np.array(tomaxvalues), np.argmax, 1, playersL)
			return maxv.pop()

		return [get('D(L)')] + [get('D(R)')]

	def _chooseDefenceCenter(self, team, opteam, num):
		players = list(getPlayersFromTeamByPos(self.teamdata, team, 'D(C)'))
		params = ['TotalTackles', 'AerialWon', 'Rating','OffsidesWon','GameStarted',\
		 'ShotsBlocked', 'LastName']
		tomaxvalues = self._getParamValues(players, params)
		tominvalues = self._getParamValues(players,['AerialLost','Dispossesed','Yellow'])
		maxv = self._optimalPlayers(np.array(tomaxvalues), np.argmax, num, players)
		minv = self._optimalPlayers(np.array(tominvalues), np.argmin, num, players)
		#if same players both in maxv and minv append in result list
		data = list(set(maxv).intersection(set(minv)))
		#self._opteamOptimal(self.fwparams, opteam, players)
		size = len(data)
		if size == 2: return data
		else: return data + list(filter(lambda x: x not in data, maxv))


	def _chooseForward(self, team, opteam, num):
		'''
			Choose best forward for this moment
		'''
		players = list(getPlayersFromTeamByPos(self.teamdata, team, 'FW'))
		if num > len(players):
			#raise OptimalTeamException("Count of selectable players, more than players")
			num = len(players)
		result = self._getParamValues(players, self.fwparams)
		target = self._optimalPlayers(np.array(result), np.argmax, num, players)
		return target

	def _optimalPlayers(self, matr, func, num, players):

		@recur.tco
		def optimalInner(matr, func, num, players, res=set()):
			c = Counter(func(matr, axis=0)).most_common(num)
			result = set(map(lambda x: players[x[0]]['LastName'], c))
			if len(result) == num: 
				res |= result
				return False, list(res)
			idxs = list(filter(lambda x: players[x]['LastName'] in result, \
				range(len(players))))
			temp = matr.tolist()
			for i in idxs:
				del players[i]
				del temp[i]
			res |= result
			return True, (np.array(temp), func, num-len(result), players, res)
		return optimalInner(matr, func, num, players, )

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


class StatisticsException(Exception):
	pass

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

	def _checkTeam(self, team):
		if team not in self.teamdata:
			raise StatisticsException('{0} not contains in teams'.format(team))

	def comparePlayers(self, data1, data2):
		'''
			Format for data1 and for data2 is tuple (team, lastname)
		'''
		team1, player1 = data1
		team2, player2 = data2
		self._checkTeam(team1)
		self._checkTeam(team2)
		playerdata1 = getPlayer(self.teamdata[team1], player1)
		playerdata2 = getPlayer(self.teamdata[team2], player2)
		result1 = 0
		result2 = 0
		print('{0} vs {1}'.format(player1, player2))
		for item in playerdata1.keys():
			try:
				if int(playerdata1[item]) > int(playerdata2[item]):
					result1 += 1
				elif int(playerdata1[item]) < int(playerdata2[item]):
					result2 += 1
				else:
					result1 += 1
					result2 += 1
			except Exception as e:
				pass
			print('{0} : {1} - {2}'.format(item, playerdata1[item], playerdata2[item]))
		print(' ')
		print('Result: {0} - {1}'.format(result1, result2))


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

