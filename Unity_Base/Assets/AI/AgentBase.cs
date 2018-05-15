using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class AgentBase : MonoBehaviour
{
    public class AgentData
    {
        public List<float> Observers { get; set; }
        public int Action { get; set; }
        public float Reward { get; set; }
    }

    public enum State
    {
        None = 0,
        Wait,
        Begin,
        Loop,
    }

    private const int SEND_DATA_SIZE = 300;
    
    private SocketManager mSocketManager = null;
    private JsonParser mJParser = new JsonParser();

    private List<AgentData> mLsAgentData = new List<AgentData>();
    private AgentData mCurAgentData = null;

    private State mState = State.None;
    private int mGlobalStep = 0;
    private int mStep = 0;

    public abstract void Init();
    public abstract List<float> DefineObservations();
    public abstract void Step(int action);
    public abstract bool DefineDone();
    public abstract float DefineReward();

    private void Awake()
    {
        mSocketManager = new SocketManager(this);
    }

    private void Update()
    {
        switch (mState)
        {
            case State.Wait:
                return;

            case State.Begin:
                Begin();
                return;

            case State.Loop:
                Loop();
                return;
        }
    }

    private void Begin()
    {
        Init();

        mGlobalStep++;
        mStep = 0;
        mLsAgentData.Clear();

        mCurAgentData = CreateAgentData();
        mCurAgentData.Observers = DefineObservations();

        mState = State.Wait;
        string jsonData = mJParser.GetActionJsonData(mCurAgentData);
        mSocketManager.SendData(jsonData);

        OnWait();
    }

    private void Loop()
    {
        mStep++;

        Step(mCurAgentData.Action);
        mCurAgentData.Reward = DefineReward();
        bool isDone = DefineDone();

        if (isDone)
        {
            print("GlobalCount : " + mGlobalStep + "  [Step : " + mStep + "]"/* + "[Reward : " + _done_GameController.scoreText.text + "]"*/);

            mState = State.Wait;
            List<AgentData> temp = new List<AgentData>();
            int size = mLsAgentData.Count;
            for (int i = 0; i < mLsAgentData.Count; ++i)
            {
                temp.Add(mLsAgentData[i]);
                if (temp.Count == SEND_DATA_SIZE || i == mLsAgentData.Count - 1)
                {
                    string trainJsonData = mJParser.GetTrainJsonData(temp, size);
                    mSocketManager.SendData(trainJsonData);
                    temp.Clear();
                }
            }
            OnWait(10000);
            return;
        }

        mCurAgentData = CreateAgentData();
        mCurAgentData.Observers = DefineObservations();

        mState = State.Wait;
        string jsonData = mJParser.GetActionJsonData(mCurAgentData);
        mSocketManager.SendData(jsonData);

        OnWait();
    }

    private AgentData CreateAgentData()
    {
        AgentData data = new AgentData();
        mLsAgentData.Add(data);

        return data;
    }

    private void OnWait(long timeout = 5000)
    {
        Stopwatch sw = new Stopwatch();
        while (mState == State.Wait)
        {
            sw.Start();
            long time = sw.ElapsedMilliseconds;
            sw.Stop();

            if (time >= timeout)
            {
                print("TimeOut : " + time);
                mState = State.None;
                break;
            }
        }
    }

    public void OnStart()
    {
        mState = State.Begin;
    }

    public void ListenToData(string data)
    {
        string result = mJParser.GetResultData(data);
        switch(result)
        {
            case "action":
                int action = mJParser.GetActionData(data);
                mCurAgentData.Action = action;
                mState = State.Loop;
                break;

            case "train":
                OnStart();
                break;
        }
    }
}