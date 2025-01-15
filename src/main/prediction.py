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
st.title("Premier Predict")

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

def venueType(team_match):
    if team_match['venue'].iloc[0] == "Home":
        return "Away"
    else:
        return "Home"


display = False


def display_results():
    if team_match.empty and not display:
        st.warning(f"No matches found for {team} in Matchweek {week}.")
    else:
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.header("Match Facts")
            st.write("Date and Time:")
            st.write("Referee:")
            st.write("Teams:")
            st.write("Home/Away:")
            st.write("Goals Scored:")
            st.write("Posession:")
            st.write("Expected Goals:")
            st.write("Formation:")
            st.write("Shots on Target:")
            st.write("Shots taken:")
        with col2:
            st.header("")
            st.write(f"{team_match['date'].iloc[0].date()}")
            st.write(f"{team_match['referee'].iloc[0]}")
            st.write(f"{team}")
            st.write(f"{team_match['venue'].iloc[0]}")
            st.write(f"{team_match['gf'].iloc[0]}")
            st.write(f"{team_match['poss'].iloc[0]}%")
            st.write(f"{team_match['xg'].iloc[0]}")
            st.write(f"{team_match['formation'].iloc[0]}")
            st.write(f"{team_match['sot'].iloc[0]}")
            st.write(f"{team_match['sh'].iloc[0]}")
        with col3:
            st.header("")
            st.write(f"{team_match['time'].iloc[0]}")
            st.write("-")
            st.write(f"{team_match['opponent'].iloc[0]}")
            st.write(venueType(team_match))
            st.write(f"{team_match['ga'].iloc[0]}")
            st.write(f"{100 - team_match['poss'].iloc[0]}%")
            st.write(f"{team_match['xga'].iloc[0]}")
            st.write(f"{team_match['opp formation'].iloc[0]}")
                        
        
        st.header(f"Predicted vs Actual result for: {team} vs {team_match['opponent'].iloc[0]}")
        st.write(f"Predicted result: {team_match['predicted_result'].iloc[0]} for {team}")
        st.write(f"Actual result: {team_match['actual_result'].iloc[0]} for {team}")
         
            
if st.button("Results",use_container_width=True):
    display = True
    display_results()

