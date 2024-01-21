using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Manager : MonoBehaviour
{
    [Range(1,20)] public int politicianCount;
    [Range(2,1000)] public int citizenCount;
    [Range(0f, 1f)] public float visibleCitizensPercentage;
    [Range(1,100)] public int issuesCount;
    [Range(0f, 1f)] public float learningRate;
    [Range(0f, 1f)] public float epsilon;
    public int epochs;
    [SerializeField] private GameObject politicianPrefab;
    private GameObject[] politicians;
    private ANN DQN;
    private float[][] politicianStanding;
    private float[] averagePoliticianStanding;
    private int[] votes;
    private Citizen[] citizens;
    private Replay replay;
    private int stateSize;
    private int actionSize;
    void Start()
    {
        stateSize = (int)(citizenCount * visibleCitizensPercentage) + issuesCount + politicianCount;
        actionSize = issuesCount * 2 + 1;
        DQN = new ANN(new int[]{stateSize, stateSize/2, stateSize/3, stateSize/4, actionSize}, new string[]{"Input", "Tanh", "Tanh", "Tanh", "Tanh"});
        InstantiateCitizens();
        InitializePoliticians();
        Emulate(epochs);

    }
    float time = 0;
    void FixedUpdate()
    {
        Election();
        UpdateObjects();
        if (time >= 5f) {
            RandomizePoliticianStances();
            RandomizeCitizenStances();
            time = 0;
        }
        time += Time.deltaTime;;
    }

    private void InitializePoliticians() {
        politicians = new GameObject[politicianCount];
        politicianStanding = new float[politicianCount][];
        for (int i = 0; i < politicianCount; i++) {
            politicianStanding[i] = new float[issuesCount];
            for (int j = 0; j < issuesCount; j++) {
                politicianStanding[i][j] = Random.Range(-1f, 1f);
            }
        }
        averagePoliticianStanding = new float[politicianCount];
        votes = new int[politicianCount];
        for (int i = 0; i < politicianCount; i++) {
            politicians[i] = Instantiate(politicianPrefab, new Vector3(i, 0, 0), Quaternion.identity);
        }
        Camera.main.orthographicSize = politicianCount/2f + 1f;
        Camera.main.transform.position = new Vector3(politicianCount / 2 + 0.5f, 0, -10);
    }

    private void InstantiateCitizens() {
        citizens = new Citizen[citizenCount];
        for (int i = 0; i < citizenCount; i++) {
            citizens[i] = new Citizen(issuesCount);
        }
    }
    private void Emulate(int N) {
        replay = new Replay(politicianCount, stateSize, 1, 1);
        Debug.Log("Started Training...");
        for (int i = 0; i < N; i++) {
            RandomizeCitizenStances();
            RandomizePoliticianStances();
            CalculateAverageStandings();
            float[][] states = new float[politicianCount][];
            float[][] actions = new float[politicianCount][];
            float[][] rewards = new float[politicianCount][];
            float[][] nextStates = new float[politicianCount][];
            for (int j = 0; j < politicianCount; j++) {
                float[] state = GetVisibleCitizens().Concat(politicianStanding[j]).Concat(averagePoliticianStanding).ToArray();
                float[][] allStates = GetAllStates(politicianStanding[j]);
                float[] outputs = DQN.OutputSoftmax(state);
                float max = outputs[0];
                int index = 0;
                for (int k = 1; k < outputs.Length; k++) {
                    if (outputs[k] > max) {
                        max = outputs[k];
                        index = k;
                    }
                }
                if (Random.Range(0f, 1f) < epsilon)
                    index = Random.Range(0, allStates.Length);
                float[] action = (float[])allStates[index].Clone();
                politicianStanding[j] = action.Clone() as float[];
                states[j] = state;
                actions[j] = new float[] {index};
                nextStates[j] = action;
            }
            int winner = GetWinner();
            for (int j = 0; j < politicianCount; j++) {
                float[] reward = new float[] {GetVotes(j)};
                if (j == winner)
                    reward[0] *= 2;
                reward[0] /= citizenCount;
                rewards[j] = reward;
            }
            for (int j = 0; j < politicianCount; j++) {
                replay.Add(states[j], actions[j], rewards[j], nextStates[j]);
            }
            TrainModel();
        }
        Debug.Log("Finished Training");
    }

    private void TrainModel() {
        for (int i = 0; i < politicianCount; i++) {
            float[][] currentValues = replay.GetValues(i);
            float[] nextState = currentValues[3];
            float[] currentState = currentValues[0];
            float[] nextOutputs = DQN.OutputSoftmax(nextState);
            float[] currentOutputs = DQN.OutputSoftmax(currentState);
            float reward = currentValues[2][0];
            float targetQVal = reward + 0.99f * nextOutputs.Max();
            float currentQVal = currentOutputs[(int)currentValues[1][0]];
            float loss = Mathf.Pow(targetQVal - currentQVal, 2);
            float[] expectedOutput = currentOutputs;
            expectedOutput[(int)currentValues[1][0]] = targetQVal;
            DQN.BackPropogate(expectedOutput, learningRate);
        }
    }

    private int GetWinner() {
        for (int i = 0; i < politicianCount; i++) {
            votes[i] = 0;
        }
        for (int i = 0; i < citizenCount; i++) {
            votes[citizens[i].Vote(politicianStanding)]++;
        }
        int winner = 0;
        for (int i = 0; i < politicianCount; i++) {
            if (votes[i] > votes[winner]) {
                winner = i;
            }
        }
        return winner;
    }
    private int GetVotes(int i) {
        int votes = 0;
        for (int j = 0; j < citizenCount; j++) {
            if (citizens[j].Vote(politicianStanding) == i) {
                votes++;
            }
        }
        return votes;
    }
    private void Election() {
        GetNewStandings();
        int winner = GetWinner();
    }
    private void GetNewStandings() {
        CalculateAverageStandings();
        float[] visibleCitizens = GetVisibleCitizens();
        for (int i = 0; i < politicianCount; i++) {
            float[] standings = politicianStanding[i].Clone() as float[];
            float[] currentAvgStandings = averagePoliticianStanding.Clone() as float[];
            float[] inputs = visibleCitizens.Concat(standings).Concat(currentAvgStandings).ToArray();
            float[] outputs = DQN.OutputSoftmax(inputs);
            float[][] allStates = GetAllStates(standings);
            float max = outputs[0];
            int maxIndex = 0;
            for (int j = 1; j < outputs.Length; j++) {
                if (outputs[j] > max) {
                    max = outputs[j];
                    maxIndex = j;
                }
            }
            float[] target = allStates[maxIndex];
            politicianStanding[i] = target;
        }
    }
    
    private void CalculateAverageStandings() {
        for (int i = 0; i < politicianCount; i++) {
            averagePoliticianStanding[i] = 0;
            for (int j = 0; j < issuesCount; j++) {
                averagePoliticianStanding[i] += politicianStanding[i][j];
            }
            averagePoliticianStanding[i] /= issuesCount;
        }
    }
    private float[] GetVisibleCitizens()
    {
        float[] visibleCitizens = new float[(int)(citizenCount * visibleCitizensPercentage)];
        ShuffleCitizens();
        for (int i = 0; i < visibleCitizens.Length; i++)
        {
            visibleCitizens[i] = citizens[i].GetAverageIssue();
        }
        return visibleCitizens;
    }

    private void ShuffleCitizens()
    {
        citizens = citizens.OrderBy(x => Random.value).ToArray();
    }

    private void UpdateObjects() {
        for (int i = 0; i < politicianCount; i++) {
            politicians[i].GetComponent<PoliticianScript>().UpdateSelf(averagePoliticianStanding[i], votes[i], i == GetWinner());
        }
    }

    private void RandomizeCitizenStances() {
        for (int i = 0; i < citizenCount; i++) {
            citizens[i].RandomizeStance();
        }
    }
    private void RandomizePoliticianStances() {
        for (int i = 0; i < politicianCount; i++) {
            for (int j = 0; j < issuesCount; j++) {
                politicianStanding[i][j] = Random.Range(-1f, 1f);
            }
        }
    }
    private float[][] GetAllStates(float[] issueStance) {
        float[][] allStates = new float[issuesCount * 2 + 1][];
        allStates[0] = (float[])issueStance.Clone();
        int index = 1;
        for (int i = 0; i < issuesCount; i++) {
            allStates[index] = (float[])issueStance.Clone();
            allStates[index][i] += 0.1f;
            allStates[index][i] = Mathf.Clamp(allStates[index][i], -1f, 1f);
            index++;
            allStates[index] = (float[])issueStance.Clone();
            allStates[index][i] -= 0.1f;
            allStates[index][i] = Mathf.Clamp(allStates[index][i], -1f, 1f);
            index++;
        }

        return allStates;
    }
}
class Citizen {
    float[] issues;
    int issuesCount;
    public Citizen(int issuesCount) {
        this.issues = new float[issuesCount];
        this.issuesCount = issuesCount;
        RandomizeStance();
    }
    private float NextGaussian() {
       float u1 = 1.0f - UnityEngine.Random.Range(0f, 1f);
        float u2 = 1.0f - UnityEngine.Random.Range(0f, 1f); 

        // Box-Muller transform
        float z0 = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);

        return Mathf.Clamp(z0, -1f, 1f);
    }

    public void RandomizeStance() {
        for (int i = 0; i < issuesCount; i++) {
            issues[i] = Random.Range(-1f, 1f);
        }
    }

    public int Vote(float[][] polticianStandings) {
        int bestPolitician = 0;
        float bestPoliticianScore = float.MaxValue;
        for (int i = 0; i < polticianStandings.Length; i++) {
            float politicianScore = 0;
            for (int j = 0; j < issuesCount; j++) {
                politicianScore += Mathf.Abs(polticianStandings[i][j] - issues[j]);
            }
            if (politicianScore < bestPoliticianScore) {
                bestPolitician = i;
                bestPoliticianScore = politicianScore;
            }
        }
        return bestPolitician;
    }


    public float GetAverageIssue() {
        float average = 0;
        for (int i = 0; i < issuesCount; i++) {
            average += issues[i];
        }
        return average / issuesCount;
    }

}

class Replay {
    float[][][] memory;
    int memoryIndex;
    public Replay(int memorySize, int stateSize, int actionSize, int rewardSize) {
        memory = new float[memorySize][][];
        memoryIndex = 0;
        for (int i = 0; i < memorySize; i++) {
            memory[i] = new float[4][];
            memory[i][0] = new float[stateSize];
            memory[i][1] = new float[actionSize];
            memory[i][2] = new float[rewardSize];
            memory[i][3] = new float[stateSize];
        }
    }

    public void Add(float[] state, float[] action, float[] reward, float[] nextState) {
        memory[memoryIndex][0] = state;
        memory[memoryIndex][1] = action;
        memory[memoryIndex][2] = reward;
        memoryIndex++;
        if (memoryIndex >= memory.Length) {
            memoryIndex = 0;
        }
    }

    public float[][] GetValues(int i) {
        return memory[i];
    }
}