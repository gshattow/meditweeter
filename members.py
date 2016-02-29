def make_ascii(s) :
	s = "".join(i for i in s if ord(i)<128)
	s = s.replace('\n', ' ')
	s = s.replace('\r', ' ')
	return s


from twitter import *
import time

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))


def query_user(sn) :
	return twitter.friends.ids(screen_name = sn)

def query_list(sn, list, cursor) :
	return twitter.lists.members(owner_screen_name=sn, slug=list, cursor=cursor)


def get_friends(sn, query, md) :
	for n in range(0, len(query["ids"]), 10):
		outfile = sn + '_labeled.tsv'
		f = open(outfile, 'a')
		if n == 0 : f = open(outfile, 'w')
		ids = query["ids"][n:n+10]

		#-----------------------------------------------------------------------
		# create a subquery, looking up information about these users
		# twitter API docs: https://dev.twitter.com/docs/api/1/get/users/lookup
		#-----------------------------------------------------------------------
		subquery = twitter.users.lookup(user_id = ids)

		for user in subquery:
			#-----------------------------------------------------------------------
			# now print out user info, starring any users that are Verified.
			#-----------------------------------------------------------------------
			if len(user["screen_name"] + make_ascii(user["name"]) + make_ascii(user["description"])) > 0 :
				f.write("%d\t%s %s %s \n" % (md, user["screen_name"], make_ascii(user["name"]), make_ascii(user["description"])))
		f.close()
		time.sleep(50)

def get_members(sn, query, md, run) :
	users = query['users']
	next_cursor = query['next_cursor']
	print next_cursor, run, len(users)
	outfile = sn + '_labeled.tsv'
	f = open(outfile, 'a')
	if run == 0 : f = open(outfile, 'w')
	for user in users:
		#-----------------------------------------------------------------------
		# now print out user info, if it exists in ascii.
		#-----------------------------------------------------------------------
		if len(user["screen_name"] + make_ascii(user["name"]) + make_ascii(user["description"])) > 0 :
			f.write("%d\t%s %s %s \n" % (md, user["screen_name"], make_ascii(user["name"]), make_ascii(user["description"])))
	f.close()
	time.sleep(50)
	run += 1
	if next_cursor != 0 : 
		query = query_list(sn, list, next_cursor)
		get_members(sn, query, md, run)
		
		
	return run

# sn = 'MrsFridayNext'
# # sn = 'VivaLaRouge'
# md = 0
# query = query_user(sn)
# get_friends(sn, query, md)

# sn = 'hrana'
# list = 'twitter-doctors'
# md = 1
# sn = 'verified'
# list = 'verified-accounts'
# md = 0
# sn = 'Scobleizer'
# list = 'most-influential-in-tech'
# md = 0
sn = 'Pierre_ADroniou'
list = 'speak-about-iot'
md = 0
query = query_list(sn, list, -1) 
run = 0
run = get_members(sn, query, md, run) 


#query = twitter.friends.ids(screen_name = sn)
#query = twitter.lists.members(owner_screen_name="hrana", slug="twitter-doctors")
#query = twitter.lists.members(owner_screen_name="verified", slug="verified-accounts")

