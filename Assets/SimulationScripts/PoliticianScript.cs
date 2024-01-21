using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PoliticianScript : MonoBehaviour
{
    public float averageVal;
    public void UpdateSelf(float averageVal, int votes, bool isWinner) {
        this.averageVal = averageVal;
        if (averageVal > 0)
            GetComponentInChildren<SpriteRenderer>().color = new Color(1 - averageVal, 0f, 0f);
        else if (averageVal < 0)
            GetComponentInChildren<SpriteRenderer>().color = new Color(0f, 0f, 1 - averageVal * -1);
        else
            GetComponentInChildren<SpriteRenderer>().color = new Color(1f, 1f, 1f);
        GetComponentInChildren<TextMeshPro>().text = votes.ToString() + "\n" + averageVal.ToString("0.00");
        if (isWinner)
            GetComponentInChildren<TextMeshPro>().color = Color.green;
        else
            GetComponentInChildren<TextMeshPro>().color = Color.black;
    }
}
