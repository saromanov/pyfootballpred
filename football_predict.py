import numpy as np
import urllib.request
import json
import itertools
import functools
from collections import Counter, namedtuple
from fn import F, op, _
from fn import recur
from fn.iters import take, drop, map, filter
import builtins
from multiprocessing import Pool, Process, Queue, Lock, Array, Value
import textblob
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import linear_model

#from requests import async

#Only for English Premier League
teamsIds = ['26','167','15','13','31','32','18','162','30','23','96','259','29','175','24',
'214','16','170','168','188']


#TODO: Fix bug with teams with two words. For example: Manchester United

#http://www.whoscored.com/Teams/32

LASTNAME = 'lastname'

def loadFromUrl(url):
	opener = urllib.request.build_opener()
	opener.addheaders = [('User-agent', 'Mozilla/5.0')]
	urllib.request.install_opener(opener)
	response = urllib.request.urlopen(url)
	return response.read().decode(response.headers.get_content_charset())

#Data parse from http://www.whoscored.com
class ManageData:
	def __init__(self, *args, **kwargs):
		url = kwargs.get('url')
		path = kwargs.get('path')
		self.data = None
		if url != None:
			self.data = self._readData(url)
		if path != None:
			self.data = self._loadData(path)
			self.data = self.transformData(self.data)
		if self.data != None:
			self.teams = self.data

	def _readData(self, url):
		return loadFromUrl(url)

	def transformData(self, data):
		""" Transform loaded data(teams) to lower case """
		if data == None:
			raise Exception("Current data was not loaded")
		teams = data['teams']
		return {
			team.lower(): \
			[{param.lower():player[param] for param in player.keys()} \
					for player in teams[team]]\
			for team in teams.keys()
			}

	#load data from json file file
	def _loadData(self, path):
		return loadfromJSON(path)

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
			newvalue = [(t[param], t[LASTNAME], t['TeamName']) for t in teamsdata[team]]
			return True, (teamsdata, teams, param, data+newvalue, )

		values = recurgetBest(self.teams, list(self.teams.keys()), param)
		if limit == None:
			limit = len(values)
		return list(reversed(sorted(values)))[0:limit]

	def getBestByTeam(self, param ,team, *args, **kwargs):
		""" Return list of tuple of (player, position in rating) by param

			contains zero - contains player which have a zero points in param 
		"""
		startdard = lambda playerteam: playerteam == team
		withoutzero = lambda data: data[2] == team and data[0] != 0
		bestresults = self.getBest(param)
		containszero = kwargs.get('containszero', True)
		return [(num+1, player[0], player[1]) for num, player in enumerate(bestresults) \
		if startdard(player[2])]

	def _preparegetBestByAllParams(self, params):
		if 'team' in params and team != None and team in self.teams:
			return team, self.teams[team]
		if 'players' in params:
			""" Return tuple """
			result = params['players']
			return result[0], \
			list(map(lambda x: getPlayer(self.teams[result[0]],x), \
				functools.reduce(list.__add__, \
					list(result[1].values()),[])
				)
			)

	def getBestByAllParams(self, *args,**kwargs):
		""" Get best player by value, ...all params
		"""
		team, params = self._preparegetBestByAllParams(kwargs)
		bestparam = 9999
		result = None
		params = list(params[0].keys())
		[params.remove(param) for param in ['Name', LASTNAME, 'FirstName',\
			'TeamName','PlayedPositionsRaw','TeamId', 'DateOfBirth','PlayerId',\
			'WSName', 'KnownName', 'IsCurrentPlayer','PositionShort',\
			'PositionText','TeamRegionCode','PositionLong']]
		for param in params:
			try:
				best = self.getBestByTeam(param, team)[0]
				if best[0] < bestparam:
					bestparam = best[0]
					result = (param, best)
			except Exception as e:
				pass
		return result


	def parseOnlineTextGame(self, url):
		'''
			http://www.whoscored.com/Matches/829535/Live
		'''
		data = self._readData(url)
		if data == None:
			raise Exception("Something went wrong in read data")
		idxstart = data.find('commentaryUpdater.load([[')
		if idxstart != -1:
			idxend = data.find(']);', idxstart)
			if idxend == -1:
				raise Exception("Something went wrong in parse Online Text Game")
			result = []
			for value in data[idxstart : idxend].split('\n'):
				preres = value[value.find('[')+2: value.find(']')]
				try:
					splitdata = preres.split(',')
					mins = int(splitdata[0].split('\\')[0])
					typeevent = splitdata[1][1:-1]
					result.append((mins, typeevent, splitdata[2]))
				except Exception as e:
					pass
			return result, self._getHeaderofGame(data)

	def _getHeaderofGame(self, data):
		'''
			Parse main information about game (title, score)
		'''
		target = 'matchHeader.load('
		startidx = data.find(target)
		if startidx != -1:
			finishidx = data.find(']', startidx)
			if finishidx == 1:
				raise Exception("Something wrong with getting results of the game")
			subdata = data[startidx + len(target)+1: finishidx]
			splitted = subdata.split(',')
			game = (splitted[2][1:-1], splitted[3][1:-1])
			score = splitted[-1:][0][1:-1]
			return game, score

	def getAllTeams(self):
		""" After load of team data, return all teams name """
		return list(self.data.keys())

	def getAllParamPlayers(self):
		""" After load of team data, return all params with each player """
		return list(self.data['chelsea'][0].keys())

	def getDataFromTeams(self, param, team):
		""" Return some param from all players, from all teams 
			One of the main func for Finder
		"""
		data = self.data
		if team != None:
			team = team.lower()
		result = []
		def getPlayer(player):
			return player[param], player[LASTNAME]
		def appendPlayers(team):
			result.extend(list(map(getPlayer, data[team])))
		if team != None and team in self.data:
			#data = self.data[team]
			appendPlayers(team)
			return PlayerData(result)
		for lteam in data:
			appendPlayers(lteam)
		return PlayerData(result)


def loadfromJSON(path):
	data = None
	try:
		f = open(path)
		data = f.read()
	except Exception as e:
		raise "File not found"
	return json.loads(data)

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
	""" Return all players in target pos.
	For example all forwards from team """
	return list(filter(lambda x: x['positionshort'] == pos, teamsdata[team]))


def getPlayer(teamdata, lastname):
	""" 
	teamdata - dict with all teams
	Return target player from team by last name """
	return list(filter(lambda x: x[LASTNAME] == lastname, teamdata))[0]

def dataToNames(data):
	""" Change list with params to only last name
	"""
	return list(map(lambda x: x[LASTNAME], data))

def getPlayersByParams(teamdatas, params):
	'''
		List of params for all players from all teams
	'''
	teams = list(teamdatas.keys())
	allparams = list(teamdatas[teams[0]][0].keys())
	for team in teams:
		for player in teamdatas[team]:
			yield (list(map(lambda y: player[y], \
				list(filter(lambda x: x in params, allparams)))))

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
		result['GK'] = self._chooseGK(team)
		result['DF'] = self._chooseDefence(team, opteam, form_res[0])
		result['MF'] = self._chooseMidfielder(team, form_res[1])
		result['FW'] = self._chooseForward(team, opteam, form_res[2])
		return result

	def _getParamValues(self, players, values):
		return list(map(lambda x: [x[p] for p in values], players))

	def _filterUsed(self, players, stored):
		""" Reject players wich alread in stored """
		return list(filter(lambda x: x[LASTNAME] not in stored, players))

	def _getTargetPlayers(self, team, num, pos, params, stored=[]):
		players = []
		if type(pos) == builtins.list:
			for p in pos:
				players += self._filterUsed(
					getPlayersFromTeamByPos(self.teamdata, team, p), stored)
			pos = pos[0]
		else:
			players = self._filterUsed(\
				getPlayersFromTeamByPos(self.teamdata, team, pos), stored)


		if len(players) == 0:
			players = self._getFromAnotherPos(team, pos, num, \
				list(map(lambda x:x[LASTNAME], players)))
		vecparams = self._getParamValues(players, params)
		matr = np.array(vecparams)
		if len(matr) > 0:
			if len(matr) > num: 
				return self._optimalPlayers(matr, np.argmax, num, players)
			if len(matr) ==  num:
				return list(map(lambda x: x[LASTNAME], players))
			else:
				another = self._getFromAnotherPos(team, pos, num, players)
				return dataToNames(another)


	def _chooseGK(self, team):
		""" Choose optimal Goalkeeper
		"""
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

		def getMFL(func, team, num, stored=[]):
			pos = ['AM(LR)', 'AM(CLR)', 'AM(R)', 'AM(L)']
			params = ['Rating', 'KeyPasses']
			return func(team, num, pos, params, stored)
		result = []
		cent = int(num)/2
		lf = num - cent
		if num == 4:
			cent = 2
			lf = 2
		if num == 5:
			cent = 3
			lf = 2
		if num == 3:
			cent = 1
			lf = 2
		center = getMFCenter(self._getTargetPlayers, team,cent)
		#Get midfielders left and right
		lr = getMFL(self._getTargetPlayers, team, lf, center)
		return center + lr

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
		""" Choose Left flang and Right Defenders

		arguments:
		team - target team
		num - number of defenders needed

		return:
		list of Optimal defenders(last names)
		"""
		result = []
		if num == 1:
			pass

		def get(idv):
			params = ['KeyPasses', 'Dribbles', 'TotalPasses', 'Rating', 'OffsidesWon', \
			'GameStarted']
			playersL = list(getPlayersFromTeamByPos(self.teamdata, team, idv))
			if len(playersL) == 0:
				'''
					In this case choose players from another position,
					for example D(R) -> D(CR)
				'''
				playersL = self._getFromAnotherPos(team, idv, num, result)

			#print("PLAYERS: ", list(map(lambda x: x['PositionShort'], self.teamdata[team])))
			tomaxvalues = self._getParamValues(playersL, params)
			maxv = self._optimalPlayers(np.array(tomaxvalues), np.argmax, 1, playersL)
			return maxv.pop()
		result.append(get('D(L)'))
		result.append(get('D(R)'))
		return result

	def _getFromAnotherPos(self, team, pos, num, stored):
		"""
		'plan b', get players from another positions
			in case when no players in the target pos.

			arguments:
			team - target team
			pos - current pos
			num - number of players needed
			stored - already chosen players

			return: list of players(raw)

			TODO: set ranking
		"""
		anotherpos = pos[0:pos.find('(')]
		return list(filter(lambda x: anotherpos in x['PositionShort'] and\
							             x[LASTNAME] not in stored
			, self.teamdata[team]))[0:num]


	def _chooseDefenceCenter(self, team, opteam, num):
		""" Get optimal Defender to center """
		players = list(getPlayersFromTeamByPos(self.teamdata, team, 'D(C)'))
		params = ['TotalTackles', 'AerialWon', 'Rating','OffsidesWon','GameStarted',\
		 'ShotsBlocked', LASTNAME]
		tomaxvalues = self._getParamValues(players, params)
		tominvalues = self._getParamValues(players,['AerialLost','Dispossesed','Yellow', LASTNAME])
		if len(tomaxvalues) > 1:
			maxv = self._optimalPlayers(np.array(tomaxvalues), np.argmax, num, players)
		else:
			maxv = tomaxvalues[0][-1:]

		if len(tominvalues) > 1:
			minv = self._optimalPlayers(np.array(tominvalues), np.argmin, num, players)
		else:
			minv = tominvalues[0][-1:]

		#if same players both in maxv and minv append in result list
		data = list(set(maxv).intersection(set(minv)))
		if len(data) < num:
			return data + list(map(lambda x: x[LASTNAME], self._getFromAnotherPos(team, 'D(C)', num-len(data), data)))
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
			result = set(map(lambda x: players[x[0]][LASTNAME], c))
			if len(result) == num: 
				res |= result
				return False, list(res)
			idxs = list(filter(lambda x: players[x][LASTNAME] in result, \
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
			#print(opplayers)
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
			result.append(list(map(lambda x: [x[first], x[second],x['gamestarted']], bans[targteam])))
		return sorted(result, key=lambda x:x[2])

	def compareTeams(self, team1, team2):
		'''
			Compare players by pos with two teams
			TODO: Implement it
		'''
		if team1 not in self.teamdata or team2 not in self.teamdata:
			raise StatisticsException("On of teams not in the base")

		result = {}
		poses = set(map(lambda x: x['positionshort'], self.teamdata[team1]))
		for pos in poses:
			players1 = list(getPlayersFromTeamByPos(self.teamdata, team1, pos))
			players2 = list(getPlayersFromTeamByPos(self.teamdata, team2, pos))
			q = Queue()
			q.put(players1)
			q.put(players2)
			p = Process(target=self._compareByPos, args=(q, ))
			p.start()
			p.join()
			result = q.get()
			print(result, ...)
		return result

	def _compareByPos(self, q):
		players2 = q.get()
		players1 = q.get()
		for p1 in players1:
			for p2 in players2:
				print(p1['yellow'])
		q.put(1)


	def _checkTeam(self, team):
		if team not in self.teamdata:
			raise StatisticsException('{0} not contains in teams'.format(team))

	def compareInner(self, p1, p2):
		@recur.tco
		def compare(values, p1, p2, arr):
			if len(values) == 0: return False, arr
			item = values.pop()
			try:
				if int(p1[item]) > int(p2[item]):
					arr.append(1)
					return True, (values, p1, p2, arr, )
				if int(p1[item]) < int(p2[item]):
					arr.append(0)
					return True, (values, p1, p2, arr, )

			except Exception as e:
				return True, (values, p1, p2, arr, )
			return True, (values, p1, p2, arr, )

		values = list(p1.keys())
		result = compare(values, p1, p2, [])
		res1 = Counter(result)
		return res1[1], res1[0]

	def comparePlayers(self, data1, data2):
		'''
			Format for data1 and for data2 is tuple (team, lastname)
			"Head to Head"
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

	def showComparePlayers(self, data1, data2):
		'''
			show/plot results in compare players
		'''
		pass

	def fit(self, Xdata, ydata, model=linear_model.LinearRegression()):
		'''
			sklearn-like style fit and predict
			Xdata - values for prediction
			ydata - labels, what we want to predict
			model - learning model (from sklearn)

			After this fit call predict from sklearn
			TODO: split on train, test and validation sets
		'''
		Xdatavalues = list(getPlayersByParams(self.teamdata, Xdata))
		ydatavalues = list(getPlayersByParams(self.teamdata, ydata))
		#linear = linear_model.LinearRegression()
		return model.fit(Xdatavalues, ydatavalues)

class Game:
	'''
		basic for game
	'''
	def __init__(self, data, info):
		self.data = data
		self.info = info


class TextGame:
	'''
		Get data from online of game
	'''
	def __init__(self, games=None):
		if isinstance(games, str):
			self.games = self._loadData(games)
		else:
			self.games = games

	def _loadData(self, path):
		return loadfromJSON(path)

	def _getRating(self, data):
		items = {'yellow card':-1, 'free kick won':2, 'goal':2, 'free kick lost':-1,\
		'miss':-1, 'red card':-1}
		return functools.reduce(lambda x,y: x+items[y[1]] if y[1] in items else x, \
			data,0)
		

	def extractInfo(self, data, targteam):
		result = list(filter(lambda x: x[2].find(targteam) != -1, data))
		rating = self._getRating(result)
		minuts = list(self._getGamesUntilMinute(25))

	#Need to append compare games in live
	def similarGames(self, current, minute, endmin=0):
		''' Проходим по всем матчам в базе и ищем наиболее похожие
			Нужна нормализация по минутам
			current - this game
			minute - until this minute
		'''
		#Get all games untill current minute
		if endmin != 0:
			targetevent = self._getBetweenMinutes(current, minute, endmin)
		else:
			targetevent = self._getGameUntilMinute(current, minute)
		events = list(self._getGamesUntilMinute(minute))
		distresult = self._distance(targetevent, events)
		#print(distresult.info, targetevent.info)
		#clusterresult = self._clustering(targetevent, events)
		return distresult

	def _clustering(self, targetgame, games):
		'''
			Find similar games with clustering
			TODO
		'''
		preparegames = list(map(lambda x: [i[1] for i in x.data], games))
		preparegame = list(map(lambda x: x[1], targetgame.data))
		lables = list(range(len(games)))
		clf = NearestCentroid()
		clf.fit(preparegames, lables)
		print(clf.predict(preparegame))

	def _distance(self, targetevent, events):
		'''
			Find similar games(events) with naive distance
			todo: normalize to similar length
			return Game object with optimal game
		'''
		SCORE_INIT = 99999
		#Global scores
		bestscore = SCORE_INIT
		bestscore2 = SCORE_INIT
		game = Game('', '')
		for event in events:
			localresult1 = 0
			localresult2 = 0
			for targ in targetevent.data:
				tmin, tdescription = targ[0], targ[1]
				preres = list(map(lambda x: -1 if x[1] != tdescription else (abs(tmin - x[0])), event.data))
				res = abs(sum(preres))
				minuscount = len(list(filter(lambda x: x == -1, preres)))
				localresult1 += minuscount
				localresult2 += res
			if localresult1 < bestscore and localresult2 < bestscore2:
				bestscore = localresult1
				bestscore2 = localresult2
				game = event
		return game

	def _getGameUntilMinute(self, game, minute):
		data, info = game
		return Game(list(reversed(list(itertools.dropwhile(lambda x: x[0] >= minute, data)))), info)

	def _getBetweenMinutes(self, game, startmin, endmin):
		""" Get game between startmin and endmin """
		data, info = game
		return Game(list(reversed(list(filter(lambda x: x[0] >= startmin and \
			x[0] <= endmin, data)))), info)

	def _getGamesUntilMinute(self, minute):
		'''
			Get games before n minutes
		'''
		if self.games != None:
			for game in self.games:
				if len(self.games[game]) > 0:
					yield self._getGameUntilMinute(self.games[game], minute)

	def getGames(self, name):
		""" Get games where contains 'name' (as team)"""
		teams = self.games.keys()
		return list(map(lambda m: self.games[m], filter(lambda x: name in x.split(), teams)))


class LiveGameAnalysisException(Exception):
	""" Exception for LiveGameAnalysis class """
	pass

class LiveGameAnalysis:
	""" 
		Analysis from life text report of the game
		TODO: Rewrite it for Finder class
	"""
	def __init__(self, *args, **kwargs):
		self.data = self._collectMatches(kwargs)
		if self.data == None:
			raise LiveGameAnalysisException("Can't load data for analysis")
		self.stat = Statistics(self.data)
		self.textgame = TextGame(self.data)

	def _collectMatches(self, param):
		""" Collect games by some param """
		return CollectMatches(param).result()

	def mostFreqEvents(self, startmin,  endmin, *args, **kwargs):
		""" Get Most frequent events from startmin until endmin 
			sample - Get sample from n games
		"""
		if( isinstance(startmin, int) != True or isinstance(endmin, int) != True):
			raise LiveGameAnalysisException("Time need to be in int format")
		if(startmin > endmin):
			raise LiveGameAnalysisException("Starttime can't be greather then endtime")
		sample = kwargs.get('sample')
		if sample != None:
			return self._sampleCase(startmin, endmin, self.data, sample)
		preresult = []
		sampledata = kwargs.get('data')
		data = self.data if sampledata == None else sampledata
		for game in data:
			if len(data[game]) > 0:
				filtering = list(
					map(lambda x: x[1], 
					filter(lambda x: x[0] >= startmin and x[0] <= endmin,\
					self.data[game][0])))
				preresult += filtering
		if len(preresult) > 0:
			limit = kwargs.get('limit',0)
			count = Counter(preresult)
			lim = len(preresult) if limit == 0 else limit
			return count.most_common()[0:lim]

	def _sampleCase(self, startmin, endmin, data, size):
		""" Return random sample with size in data
			with 
		"""
		sto = list(map(lambda x: data[x], data))
		sampledata = np.random.choice(sto,size)
		result = []
		for game in sampledata:
			if len(game) > 0:
				result += list(
						map(lambda x: x[1], 
						filter(lambda x: x[0] >= startmin and x[0] <= endmin,\
						game[0])))
		return Counter(result).most_common()

	def _findGameByTitle(self, title):
		""" Find game by title
			For example Arsenal - Chelsea
			TODO: Fix search
		"""
		prepared = title.split('-')
		firstteam = prepared[0][:-1]
		secondteam = prepared[1][1:]
		#Pretty bad solution
		for dat in self.data:
			target = self.data[dat]
			if len(target) > 0:
				teams = self.data[dat][1][0]
				if firstteam == teams[0] and secondteam == teams[1]:
					return self.data[dat]

	def similarGames(self, title, startmin, endmin):
		""" Find similar games with TextGame class """
		if startmin > endmin:
			raise LiveGameAnalysisException("Starttime can't be geather then endmin")
		result = self._findGameByTitle(title)
		self.textgame.similarGames(result, startmin, endmin=endmin)

	def CountTargetEvent(self, event, startmin, endmin=0):
		""" Count some event in every games in the base
			For example: event - 'miss' and count it in every game from
			startmin until evndmin (or just with startmin)

			output: games with number of target events

			events:
			-yellow card
			-miss
			-offside
			-free kick won
			-corner
			...
		"""
		result = list(self._countTargetEventHelp(event, startmin, endmin))
		return result

	def _countTargetEventHelp(self, event, startmin, endmin):
		for name in self.data:
			evt = self.data[name]
			if len(evt) > 0:
				if endmin == 0 and startmin != 0:
					#Not tested!
					count_events = len(list(lambda x: x[1] ==  event and \
						x[0] <= startmin, evt[0]))
				count_events = len(list(filter(lambda x: x[1] == event, evt[0])))
				yield(evt[1], count_events)

	def findGame(self, title):
		""" Another implementation of finding name by title
			On the input string representation of game
			For example Manchester United - Arsenal
		"""
		return self._findGameByTitle(title)

	def _getEventsInner(self, eventname, data=None):
		'''
			Return all events (for example "goal") from all games
			getEvents('goal')
		'''
		contain = self._prepareData(data)
		return self._mainLoop(contain, lambda x: x[1] == eventname)

	def getEvents(self, eventname, data=None):
		return list(self._getEventsInner(eventname, data=data))

	def getEventsByTime(self, startmin, endmin, data=None):
		"""
			data - Optional element. already prepared elements. No need check self.data
		"""
		if startmin < 0 or startmin > 90:
			raise Exception("Startmin can't be less than zero or greather than 90")
		if endmin < 1 or endmin > 90:
			raise Exception("Startmin can't be less than 1 or greather than 90")
		contain = self._prepareData(data)
		return self._mainLoop(contain, lambda x: x[0] >= startmin and x[0] <= endmin)

	def _prepareData(self, data):
		if data == None: return [self.data[x] for x in self.data if len(self.data[x]) > 0]
		#Already prepared
		else: return data

	def _mainLoop(self, data, func):
		'''
			After prepare data - run mainLoop over all games
		'''
		for ds in data:
			yield ds[1], self._innerFilter(func, ds[0])

	def _innerFilter(self, func, elements):
		""" 
			Filtering elements from datastore(self.data) or from already prepared (data)
		"""
		return list(filter(func, elements))


	def getAllEventsName(self):
		""" Return just all events name 
			Now is dirty solution, return event from only one game
		"""
		tempkeys = list(self.data.keys())
		return set(map(lambda x: x[1], self.data[tempkeys[0]][0]))

	def getGameWithMostFreqEvents(self):
		""" Return (number of events, game with the largets number of events)
			For example:
			Game 1:
			1 min - Cornor
			5 min - Yellow card
			85 min - Goal
			Game 2
			1 min - Cornor
			2 min - Goal
			3 min - Goal
			4 min - Goal

			Return Game 2
		"""
		return max(((len(title), title) for i, title in enumerate(self.data)), key=lambda x: x[0])



class MatchesData:
	""" Object class for matches """
	def __init__(self):
		self.result = []

	def add(self, game):
		self.result.append(game)

class PlayerData:
	def __init__(self, data):
		self.data = data

class FinderHelpful:
	def __init__(self, teams, matches):
		self.teams = teams
		self.matches = matches
		self.use = None


class QueryData(object):
	"""Set information for each query"""
	def __init__(self, arg, *args, **kwargs):
		super(QueryData, self).__init__()
		self.lastquery = kwargs.get('lastquery', None)

		

COMPLEX_QUERY = 'ComplexQuery'
SIMPLE_QUERY = 'SimpleQuery'

PLAYER_EVENT = 'playerevent'
GAME_EVENT = 'gameevent'

class Finder:
	""" Find games with natural language
		Fox example "interesting game" or "game with many yellow cards"
		TODO
		Finder("yellow cards").ident(>5)

		data can be just a query or in the map view
		map can be in key with:
		Player, Event
		For example {player: rooney}
					{event: 'miss'}

		High-Oreder class over all in this file
	"""
	def __init__(self, data, *args, **kwargs):
		self.calculation = None
		query_type = self._identQuery(data)
		self.reserved = ['player', 'event']
		preresult = kwargs.get('preresult')
		#All queries durning session
		self.queries = kwargs.get('queries', [data, 'lastname'])
		manage = ManageData(path='../teams')
		#Types for finding params on players
		self._playerParams = manage.getAllParamPlayers()
		#Types for finding params on teams
		self._teamsName = manage.getAllTeams()
		if preresult == None:
			""" Load basic classes """
			if query_type == COMPLEX_QUERY:
				""" Case for COMPLEX QUERY """
				self._complexQuery(data)
			else:
				data = self._prepareQuery(data)
				teams = manage.data
				lga = LiveGameAnalysis(data='./matches')
				self.lga = lga
				self._gameevents = lga.getAllEventsName()
				self.data = FinderHelpful(manage, lga)
				if data in self._playerParams:
					team = kwargs.get('team', None)
					self.calculation = self._asyncCall(self.data.teams.getDataFromTeams, \
						params=(data,team,))
					self.data.use = teams
					self.resultdata = self.calculation.get().data
					self.useddata = teams
					self.preresult = self.resultdata
					self.typeevent = PLAYER_EVENT
				elif data in self._teamsName:
					""" TODO: Implement it """
					pass
				elif data in self._gameevents:
					"""
						In the case in data(query) from events from game 
					"""
					self.calculation = self._asyncCall(self.data.matches.getEvents, \
					params=(data,))
					self.resultdata = self.calculation.get()
					self.preresult = self.resultdata
					self.useddata = teams
					self.typeevent = GAME_EVENT
					print("This is game event: ")
				else:
					raise Exception("This query not contain in any db")

			'''self.calculation = self._asyncCall(self.data.matches.getEvents, \
				params=('miss',))'''
		else:
			"""
				Inner call
			"""
			self.resultdata = preresult
			#Used data in this session(player data, match data)
			self.useddata = kwargs.get('useddata')
			self.preresult = preresult
			self.data = self.useddata
			self.typeevent = kwargs.get('typeevent')

		#self._findData(data)

	def _complexQuery(self, data):
		""" Just for single key """
		keys = list(data.keys())[0]
		self.target = data[keys]
		if keys == 'player':
			self._playerParams = manage.getAllParamPlayers()
		elif keys == 'event':
			self._teamsName = manage.getAllTeams()

	def _prepareQuery(self, data):
		""" Some preparation of query """
		return data.lower()

	def _identQuery(self, data):
		""" First, ident the type of query. It can be as {player: rooney} its
			a complex query or just a "rooney" its a simple query
		"""
		if type(data) ==  builtins.dict:
			return 'ComplexQuery'
		if type(data) == builtins.str:
			return 'SimpleQuery'

	def _asyncCall(self, func, params):
		pool = Pool(processes=2)
		return pool.apply_async(func, args=params)

	def _findData(self, data):
		pass

	def query(self, value,*args,**kwargs):
		if value == None:
			raise Exception("This query is empty")
		if self.calculation != None:
			matches = self.calculation.get()
			if isinstance(matches, MatchesData):
				afunc = lambda results: list(filter(lambda game: value(len(game)), \
					results))
				result = self._asyncCall(afunc, params=matches.result)
				return result
			if isinstance(matches, PlayerData):
				afunc = lambda results, val: list(filter(lambda x: val(x[0]), \
					results))
				return afunc(matches.data, value)
				#return self._asyncCall(afunc, params=(matches.data,value, )).get()
		else:
			print("WHEN CALCULATION IS ZERO", self.preresult)

	def get(self, data):
		pass

	def ident(self, param,**kwargs):
		return self._templatePred(param, lambda x: x == param)

	def greater(self, param,**kwargs):
		return self._templatePred(param, lambda x: x > param,**kwargs)

	def less(self, param,**kwargs):
		return self._templatePred(param, lambda x: x < param)

	def _templatePred(self, param, pred, **kwargs):
		""" Template for greather, less, ident and for others
			with predicate
		"""
		issort = kwargs.get('sort')
		paramvalue = self.query(pred)
		if issort:
			paramvalue = self._toSort(paramvalue)
		if self.data.use == None:
			raise Exception("Something went wrong. ")
		return Finder(param, preresult=paramvalue, useddata=self.data.use, queries=self.queries)


	def _toSort(self, paramvalue):
		""" Sorted output values """
		return sorted(paramvalue, key=lambda x: x[0], reverse=True)

	def event(self, query):
		if query in self._playerParams:
			teams = list(self.data.teams.keys())
			for team in teams:
				info = self.data.teams[team]
				result = list(filter(lambda x: x[LASTNAME] == self.target, info))
				if len(result) > 0:
					return result[0][query]
		return None

	def viewBy(self, value):
		""" View results with some value
			For example: 
				Finder('Goals')
					.greater(10)
					.viewBy(LASTNAME)
			Return pairs (goals, LastName) with greater than 10

				Finder('Goals')
					.greater(10)
					.viewBy('yellow cards')
					.viewBy(LASTNAME)
				Return list of [goals, yellow cards, LastName]
		"""
		#Now in player case
		result = []
		vbhelp = ViewByHelpful(self.preresult, self.useddata, value)
		if self.typeevent == PLAYER_EVENT:
			result = vbhelp.runPlayer()
		if self.typeevent == GAME_EVENT:
			result = vbhelp.runGame(self.preresult)
		if result != None and len(result) == 0:
			result = self.preresult
		self.queries.append(value)
		return Finder(value, preresult=result, useddata=self.useddata, queries=self.queries, typeevent=PLAYER_EVENT)

	def show(self, by=None):
		""" Output results 
			by - return only target column
		"""
		if by == None:
			return self.resultdata
		idx = self.queries.index(by)
		return list(map(lambda x: x[idx], self.resultdata))

	def between(self, startmin, endmin):
		print(self.lga)
		return Finder(startmin, preresult=self.resultdata)


class ViewByHelpful:
	def __init__(self, preresult, useddata, value):
		self.preresult = preresult
		self.useddata = useddata
		self.value = value

	def ifParamsNotEmpty(self):
		''' Before run, check if preresult and other params is not empty.
			This is checkk need to raise user exception. Of course, before 
			running run, this function call also.
		'''
		params = [self.preresult, self.useddata, self.value]
		if not all(params):
			raise Exception("Some of params, contain empty element")

	def runPlayer(self):
		self.ifParamsNotEmpty()
		findlastname = lambda name: list(filter(lambda x: name in x, self.preresult))
		result = []
		dictresult =[]
		value = self.value
		for team in self.useddata.keys():
			for player in self.useddata[team]:
				target = findlastname(player[LASTNAME])
				if len(target) > 0:
					res = list(target[0])
					if value in player:
						dictdata = {}
						dictdata[value] = player[value]
						dictdata['lastname'] = player[LASTNAME]
						res.append(player[value])
						result.append(res)
						dictresult.append(dictdata)
		return result 

	def runGame(self, gameobj):
		pass
		#print("This is preresult: ", self.preresult, self.value)

class CollectMatches:
	""" Collect all matches from web"""
	def __init__(self, param):
		""" Or or preloaded data """
		self.data=None
		if 'url' in param:
			path = './matches'
			self.iddata = list(self._parseData(param['url']))[1:]
			self.output(path)
			self.data = self._loadMatches(path)
		elif 'data' in param:
			self.data = self._loadMatches(param['data'])

	def _parseData(self, url):
		if url != None:
			result = loadFromUrl(url)
			target = "DataStore.prime('standings'"
			subpos = result.find(target)
			if subpos != -1:
				rawdata = result[subpos + len(target)+21
				: result.find(']);', subpos)]
				res = 0
				for idvalue in rawdata.split('id="'):
					splitter = idvalue.split('" ')
					titledata = "'title="
					titlesplitter = splitter[1].split('/>')[0]
					ident = titlesplitter[len(titledata):-1].split()
					if len(ident) > 0:
						yield ((ident[0], ident[2], ident[1]), idvalue.split('" ')[0])

	def result(self):
		return self.data

	def _loadMatches(self, path):
		return json.loads(open(path, 'r').read())

	def output(self, path):
		""" Output collected data at the path in pretty format
		"""
		mandata = ManageData()
		constructurl = lambda num: 'http://www.whoscored.com/Matches/{0}/LiveOld/'\
									.format(num)
		iddata = self.iddata[:3]
		resultsata = {}
		resultsata['games'] = {}
		if self.iddata != None:
			restricted = self.iddata[0:15]
			for idvalue in restricted:
				resultsata[' '.join(idvalue[0])] = mandata.parseOnlineTextGame(constructurl(idvalue[1]))
		with open(path,'w') as outfile:
			json.dump(resultsata, outfile)

def getData():
	manage = ManageData(path='../teams')
	teamdata = manage.data['teams']

def getRandomTeams(teamdata):
	teams = list(teamdata.keys())
	team2, team1 = set(np.random.choice(teams,2))
	ot = OptimalTeam(teamdata)
	result1 = ot.getOptimalTeam(team2, team1, '4-4-2')
	result2 = ot.getOptimalTeam(team1, team2, '4-4-2')
	return ((team2, result1), (team1, result2))

def getBests(teamdata):
	pass

def GkToForward(player, gk):
	''' Соотношение удара по воротам и отбитым мячам'''
	if gk[0] == 0:
		return 0
	return player[0]/gk[0]
