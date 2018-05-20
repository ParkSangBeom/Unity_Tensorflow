using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SampleAgent : AgentBase
{
    public GameObject playerObj = null;

    /// <summary>
    /// Initialization. (Restart)
    /// </summary>
    public override void Init()
    {
        playerObj.transform.position = Vector3.zero;
    }

    /// <summary>
    /// Determine the end of the game. 
    /// </summary>
    public override bool DefineDone()
    {

        float x = playerObj.transform.position.x;
        if(Mathf.Abs(x) < 3.0f)
        {
            // Continue.
            return false;
        }
        else
        {
            // End.
            return true;
        }
    }

    /// <summary>
    /// Set the observation value.
    /// </summary>
    public override List<float> DefineObservations()
    {
        List<float> lsOb = new List<float>();
        float x = playerObj.transform.position.x;
        float y = playerObj.transform.position.y;
        float z = playerObj.transform.position.z;
        lsOb.Add(x);
        lsOb.Add(y);
        lsOb.Add(z);

        return lsOb;

    }

    /// <summary>
    /// Set the reward value.
    /// </summary>
    public override float DefineReward()
    {
        float reward = 0.0f;
        float x = playerObj.transform.position.x;

        if (x > 2.9f)
        {
            reward = 1.0f;
        }
        else if(x < -2.9f)
        {
            reward = -1.0f;
        }

        return reward;
    }

    /// <summary>
    /// Set the step value.
    /// </summary>
    public override void Step(int action)
    {
        float dir = action == 0 ? -1 : 1;
        playerObj.transform.Translate(dir, 0.0f, 0.0f);
    }
}
