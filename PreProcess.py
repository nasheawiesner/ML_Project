import json
import re
import os
import glob
import scipy
from sklearn.preprocessing import normalize
import numpy as np

def vectorize_day(day):
    if day == "Sun":
        day = 0
    elif day == "Mon":
        day = 1
    elif day == "Tue":
        day = 2
    elif day == "Wed":
        day = 3
    elif day == "Thurs":
        day = 4
    elif day == "Fri":
        day = 5
    elif day == "Sat":
        day = 6
    return day

def parse():
    #path = "./formatted_data/"
    possessions = []
    labels = []
    #for file in glob.glob(os.path.join(path, '*.json')):
    with open("test.json", encoding='utf-8') as input_file:
        data = json.loads(input_file.read())

        for i in data:  #iterate through games
            player_point_count = []
            totalPointsPlayed = 0
            if 'pointsJson' in i:
                for j in i['pointsJson']: #iterate through events
                    if j["events"][0]["type"] != "Cessation" and 'line' in j:
                        for player in j['line']:
                            if player not in player_point_count:
                                player_point_count.extend([player,int(1)])
                            else:
                                player_point_count[player_point_count.index(player) + 1] += 1
                        first = True
                        possession = []
                        poss_count = 0
                        numPasses = 0
                        numTouches = 0
                        distinctPlayers = []
                        totalTime = 0
                        ours = 0
                        theirs = 0
                        totalPoints = 0
                        wind = 0
                        time = 0
                        date = 0
                        day = ""
                        lineType = 0
                        pull_start = 0
                        scored = 0.0
                        for k in j['events']:  #iterate through actions
                            index = j['events'].index(k)
                            if k['type'] == "Offense":
                                numPasses += 1
                                if k["action"] == "Goal":
                                    scored = 1.0
                                if k['passer'] not in distinctPlayers and k['passer'] != "Anonymous":
                                    distinctPlayers.append(k['passer'])
                                    numTouches += 1
                                if k['receiver'] not in distinctPlayers and k['receiver'] != "Anonymous":
                                    distinctPlayers.append(k['receiver'])
                                    numTouches += 1
                                if i["pointsJson"][-1]['summary']['lineType'] == "O":
                                    lineType = 1
                                    if first == True:
                                        pull_start = 1
                                else:
                                    lineType = 0
                                    pull_start = 0
                                if j['events'][(index + 1) % len(j['events'])]['type'] != "Offense" or index == (len(j['events']) - 1):  #if the ball gets transferred to other team the next action or it's the end of the point
                                    possession = []
                                    poss_count += 1
                                    totalPointsPlayed = 0
                                    for player in j["line"]:
                                        totalPointsPlayed += player_point_count[player_point_count.index(player) + 1]
                                    #ours = int(j['summary']['score']['ours'])
                                    #theirs = int(j['summary']['score']['theirs'])
                                    #totalPoints = ours + theirs
                                    #wind = int(i['wind']['mph'])
                                    #time = i['time']
                                    #time = int(time.replace(":", ""))
                                    #date = i["date"]
                                    #day = date.split(",")
                                    #day = vectorize_day(day[0])
                                    #date = re.findall('\d+', date)
                                    #date = int(date[0] + date[1])
                                    possession.extend([numPasses,numTouches,totalPointsPlayed, lineType, pull_start, poss_count,scored])
                                    #print(possession)
                                    possessions.append(possession)
                            elif k['type'] == "Defense":
                                possession = []
                                numPasses = 0
                                numTouches = 0
                                distinctPlayers = []
                                totalTime = 0
                                ours = 0
                                theirs = 0
                                totalPoints = 0
                                wind = 0
                                time = 0
                                date = 0
                                day = ""
                                lineType = 0
                                pull_start = 0
                            first = False
        for p in possessions:  #separate labels from training examples
            labels.append(p[-1])
            del p[-1]
        possessions = possessions / np.linalg.norm(possessions)
        labels = np.asarray(labels)
    return possessions,labels




