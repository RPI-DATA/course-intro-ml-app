from  twitter import *
import datetime, traceback 
import json
import time
import sys

def create_twitter_auth(cf_t):
        """Function to create a twitter object
           Args: cf_t is configuration dictionary. 
           Returns: Twitter object.
            """
        # When using twitter stream you must authorize.
        # these tokens are necessary for user authentication
        # create twitter API object

        auth = OAuth(cf_t['access_token'], cf_t['access_token_secret'], cf_t['consumer_key'], cf_t['consumer_secret'])

        try:
            # create twitter API object
            twitter = Twitter(auth = auth)
        except TwitterHTTPError:
            traceback.print_exc()
            time.sleep(cf_t['sleep_interval'])
        return twitter
        
def get_profiles(twitter, names, cf_t):
    """Function write profiles to a file with the form *data-user-profiles.json*
       Args: names is a list of names
             cf_t is a list of twitter config
       Returns: Nothing
        """
    # file name for daily tracking
    dt = datetime.datetime.now()
    fn = cf_t['data']+'/profiles/'+dt.strftime('%Y-%m-%d-user-profiles.json')
    with open(fn, 'w') as f:
        for name in names:
            print("Searching twitter for User profile: ", name)
            try:
                # create a subquery, looking up information about these users
                
                profiles = twitter.users.lookup(screen_name = name)
                sub_start_time = time.time()
                for profile in profiles:
                    print("User found. Total tweets:", profile['statuses_count'])
                    # now save user info
                    f.write(json.dumps(profile))
                    f.write("\n")
                sub_elapsed_time = time.time() - sub_start_time;
                if sub_elapsed_time < cf_t['sleep_interval']:
                    time.sleep(cf_t['sleep_interval'] + 1 - sub_elapsed_time)
            except TwitterHTTPError:
                traceback.print_exc()
                time.sleep(cf_t['sleep_interval'])
                continue
    f.close()
    return fn