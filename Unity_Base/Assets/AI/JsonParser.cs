using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LitJson;
using System.Text;

public class JsonParser
{
    public string GetActionJsonData(AgentBase.AgentData agentData)
    {
        StringBuilder sb = new StringBuilder();
        JsonWriter jsonw = new JsonWriter(sb);
        jsonw.WriteObjectStart();
        {
            jsonw.WritePropertyName("type");
            jsonw.Write("action");

            jsonw.WritePropertyName("datalist");
            jsonw.WriteArrayStart();
            {
                jsonw.WriteObjectStart();
                {
                    List<float> lsData = agentData.Observers;
                    for (int i = 0; i < lsData.Count; ++i)
                    {
                        jsonw.WritePropertyName("ob_" + i.ToString());
                        jsonw.Write(lsData[i]);
                    }
                }
                jsonw.WriteObjectEnd();
            }
            jsonw.WriteArrayEnd();
        }
        jsonw.WriteObjectEnd();

        return sb.ToString();
    }

    public string GetTrainJsonData(List<AgentBase.AgentData> lsAgentData, int size)
    {
        StringBuilder sb = new StringBuilder();
        JsonWriter jsonw = new JsonWriter(sb);
        jsonw.WriteObjectStart();
        {
            jsonw.WritePropertyName("type");
            jsonw.Write("train");

            jsonw.WritePropertyName("size");
            jsonw.Write(size);

            jsonw.WritePropertyName("datalist");
            jsonw.WriteArrayStart();
            {
                for (int i = 0; i < lsAgentData.Count; ++i)
                {
                    jsonw.WriteObjectStart();
                    {
                        AgentBase.AgentData agentData = lsAgentData[i];
                        for (int k = 0; k < agentData.Observers.Count; ++k)
                        {
                            jsonw.WritePropertyName("ob_" + k.ToString());
                            jsonw.Write(agentData.Observers[k]);
                        }

                        jsonw.WritePropertyName("action");
                        jsonw.Write(agentData.Action);

                        jsonw.WritePropertyName("reward");
                        jsonw.Write(agentData.Reward);
                    }
                    jsonw.WriteObjectEnd();
                }
            }
            jsonw.WriteArrayEnd();
        }
        jsonw.WriteObjectEnd();

        return sb.ToString();
    }

    public string GetResultData(string data)
    {
        JsonData jsonData = JsonMapper.ToObject(data);
        string result = jsonData["result"].ToString();
        return result;
    }

    public int GetActionData(string data)
    {
        JsonData jsonData = JsonMapper.ToObject(data);
        int value = int.Parse(jsonData["value"].ToString());
        return value;
    }
}
