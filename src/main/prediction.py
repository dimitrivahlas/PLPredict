import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import streamlit as st


url = "https://raw.githubusercontent.com/dimitrivahlas/PLPredict/main/src/main/matches.csv"
matches = pd.read_csv(url, index_col=0)

## Prepare data by putting objects to int or float so model can undestand
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

##Training model and predicting
rf = RandomForestClassifier(n_estimators = 100, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2024-08-08']
test = matches[matches["date"] > '2024-08-08']
predictors = ["venue_code", "opp_code","hour","day_code"]
rf.fit(train[predictors],train["target"])
preds = rf.predict(test[predictors]) 

##Test accuracy score
acc = accuracy_score(test["target"],preds)
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

## Test precision score
precision_score(test["target"],preds)
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf","ga","sh","sot","dist","fk","pk","pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])


def make_predictions(data, predictors):
    train = data[data["date"] < '2024-08-08']
    test = data[data["date"] > '2024-08-08']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Nottingham Forest":"Nott'ham Forest", "Sheffield United": "Sheffield Utd", "Tottenham Hotspur": "Tottenham", "West Bromwich Albion": "West Brom" ,"West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)
                
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()
test["predicted"] = preds

##Ui part
st.title("Premier League Match Predictor")

# Premier League Teams
premierLeagueTeams = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton and Hove Albion",
                      "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leicester City", "Liverpool",
                      "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest",
                       "Southampton", "Tottenham Hotspur",
                      "West Ham United", "Wolverhampton Wanderers", "Ipswich Town"]
weeks = list(range(1,21))



# Input fields for user prediction
team = st.selectbox("Select team:", premierLeagueTeams)
week = st.selectbox("Select week:",weeks)

team_match = test[(test['team'] == team) & (test['round'] == f"Matchweek {week}") & (test['date'] >= '2024-01-01')]


result_map = {1 : "Win", 0 : "Draw/Lose", "D" : "Draw", "L" : "Lose", "W": "Win" }
team_match['predicted_result'] = team_match['predicted'].map(result_map)
team_match['actual_result'] = team_match['result'].map(result_map)

if team_match.empty:
    st.warning(f"No matches found for {team} in Matchweek {week}.")
    
else:
    
    st.write(team_match['team'],  team_match['opponent'])
    
    st.write(team_match['actual_result'],team_match['predicted_result'])
    st.metric("Goals Scored", team_match['gf'])
    st.metric("Goals Conceded", team_match['ga'])
    st.metric("Shots on Target", team_match['sot'])
    

