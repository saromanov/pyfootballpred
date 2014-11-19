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
#from requests import async

#Only for English Premier League
teamsIds = ['26','167','15','13','31','32','18','162','30','23','96','259','29','175','24',
'214','16','170','168','188']


#TODO: Fix bug with teams with two words. For example: Manchester United

#http://www.whoscored.com/Teams/32

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
		if self.data != None:
			self.teams = self.data['teams']

	def _readData(self, url):
		return loadFromUrl(url)


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
		print(team, params, ...)
		bestparam = 9999
		result = None
		params = list(params[0].keys())
		[params.remove(param) for param in ['Name', 'LastName', 'FirstName',\
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
	return list(filter(lambda x: x['PositionShort'] == pos, teamsdata[team]))


def getPlayer(teamdata, lastname):
	""" 
	teamdata - dict with all teams
	Return target player from team by last name """
	return list(filter(lambda x: x['LastName'] == lastname, teamdata))[0]

def dataToNames(data):
	""" Change list with params to only last name
	"""
	return list(map(lambda x: x['LastName'], data))


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
		return list(filter(lambda x: x['LastName'] not in stored, players))

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
				list(map(lambda x:x['LastName'], players)))
		vecparams = self._getParamValues(players, params)
		matr = np.array(vecparams)
		if len(matr) > 0:
			if len(matr) > num: 
				return self._optimalPlayers(matr, np.argmax, num, players)
			if len(matr) ==  num:
				return list(map(lambda x: x['LastName'], players))
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
							             x['LastName'] not in stored
			, self.teamdata[team]))[0:num]


	def _chooseDefenceCenter(self, team, opteam, num):
		""" Get optimal Defender to center """
		players = list(getPlayersFromTeamByPos(self.teamdata, team, 'D(C)'))
		params = ['TotalTackles', 'AerialWon', 'Rating','OffsidesWon','GameStarted',\
		 'ShotsBlocked', 'LastName']
		tomaxvalues = self._getParamValues(players, params)
		tominvalues = self._getParamValues(players,['AerialLost','Dispossesed','Yellow', 'LastName'])
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
			return data + list(map(lambda x: x['LastName'], self._getFromAnotherPos(team, 'D(C)', num-len(data), data)))
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
			result.append(list(map(lambda x: [x[first], x[second],x['GameStarted']], bans[targteam])))
		return sorted(result, key=lambda x:x[2])

	def compareTeams(self, team1, team2):
		'''
			Compare players by pos with two teams
			TODO: Implement it
		'''
		if team1 not in self.teamdata or team2 not in self.teamdata:
			raise StatisticsException("On of teams not in the base")

		result = {}
		poses = set(map(lambda x: x['PositionShort'], self.teamdata[team1]))
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
		return result

	def _compareByPos(self, q):
		players2 = q.get()
		players1 = q.get()
		for p1 in players1:
			for p2 in players2:
				pass
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


	def predict(self, params, predvalue):
		'''
			params - data for prediction
			predvalue - prediction value
			Find optimal prediction
		'''
		pass

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
		self.games = games

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
	def similarGames(self, current, data, team, minute):
		''' Проходим по всем матчам в базе и ищем наиболее похожие
			Нужна нормализация по минутам
			current - this game
			data - target data
			team - this team
			minute - until this minute
		'''
		#Get all games untill current minute
		targetevent = self._getGameUntilMinute(current, minute)
		events = list(self._getGamesUntilMinute(minute))
		distresult = self._distance(targetevent, events)
		#print(distresult.info, targetevent.info)
		clusterresult = self._clustering(targetevent, events)

	def _clustering(self, targetgame, games):
		'''
			Find similar games with clustering
		'''
		preparegames = list(map(lambda x: [i[1] for i in x.data], games))
		preparegame = list(map(lambda x: x[1], targetgame.data))
		lables = list(range(len(games)))
		print(preparegame, lables)
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

	def _getGamesUntilMinute(self, minute):
		'''
			Get games before n minutes
		'''
		if self.games != None:
			for game in self.games:
				yield self._getGameUntilMinute(game, minute)


class LiveGameAnalysisException(Exception):
	""" Exception for LiveGameAnalysis class """
	pass

class LiveGameAnalysis:
	""" Analysis from life text report of the game """
	def __init__(self, *args, **kwargs):
		self.data = self._collectMatches(kwargs)
		if self.data == None:
			raise LiveGameAnalysisException("Can't load data for analysis")
		self.stat = Statistics(self.data)

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
		datagames = list(filter(lambda x: len(self.data[x]) > 0 \
			and self.data[x][1][0][0] == prepared[0] \
			and self.data[x][1][0][1] == prepared[1],\
			self.data))
		return datagames

	def similarGames(self, startmin, endmin):
		""" Find similar games with TextGame class """
		self._findGameByTitle('Arsenal-Manchester City')


class Finder:
	""" Find games with natural language
		Fox example "interesting game" or "game with many yellow cards"
		TODO
		Finder("yellow cards").ident(>5)
	"""
	def __init__(self, data, findclass=None):
		self.data = data
		if findclass != None:
			pass

	def query(self, value):
		if len(value) == 0:
			raise Exception("THis query is empty")
		return Finder(data, value)

	def ident(self, value):
		""" can be >,<,=,>=,<= """
		pass


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
