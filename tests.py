import unittest
import football_predict
#import imporfootball_predict

class TestFinderWithTeam(unittest.TestCase):
	def test_arsenal(self):
		fnd = football_predict.Finder('yellow', team='Arsenal').greater(7).viewBy('goal').viewBy('red').show()
		self.assertEqual(len(fnd), 2)
	def test_astonvilla(self):
		fnd1 = football_predict.Finder('yellow', team='aston Villa').greater(7).viewBy('goal').viewBy('red').show()
		self.assertEqual(len(fnd1), 1)
		fnd2 = football_predict.Finder('goals', team='aston Villa').greater(7).show()
		self.assertEqual(len(fnd2), 1)
	def test_chelsea(self):
		fnd = football_predict.Finder('goals', team='chelsea').greater(70).show()
		self.assertEqual(len(fnd), 0)
if __name__ == '__main__':
	unittest.main()