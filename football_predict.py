import numpy as np
import random
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
			#game, score = self._getHeaderofGame(data)
			#print(result, self._getHeaderofGame(data))
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
		#print(list(map(lambda x:x['PositionShort'], self.teamdata[team])))
		players = []
		if type(pos) == builtins.list:
			for p in pos:
				players += getPlayersFromTeamByPos(self.teamdata, team, p)
		else:
			players = list(getPlayersFromTeamByPos(self.teamdata, team, pos))

		#params = ['TotalClearances', 'Rating', 'GameStarted', 'ManOfTheMatch', 'AerialWon']
		vecparams = self._getParamValues(players, params)
		matr = np.array(vecparams)
		if len(matr) > 0: return self._optimalPlayers(matr, np.argmax, num, players)

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

		def getMFL(func, team, num):
			pos = ['AM(LR)', 'AM(CLR)', 'AM(R)', 'AM(L)']
			params = ['Rating', 'KeyPasses']
			return func(team, num, pos, params)
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
		lr = getMFL(self._getTargetPlayers, team, lf)
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
			print(result)
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


def getRandomTeams():
	manage = ManageData(path='../teams')
	teams = list(manage.data['teams'].keys())

