import React, { useState } from "react";
import axios from "axios";
import API_BASE_URL from "./api";

import { Radar, Doughnut, Line } from "react-chartjs-2";

import {
  Chart as ChartJS,
  RadialLinearScale,
  ArcElement,
  PointElement,
  LineElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Filler
} from "chart.js";

ChartJS.register(
  RadialLinearScale,
  ArcElement,
  PointElement,
  LineElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Filler
);



// CENTER TEXT PLUGIN
const centerTextPlugin = {
  id: "centerText",
  beforeDraw(chart){
    const {width,height,ctx} = chart;
    ctx.restore();

    const value = chart.config.data.datasets[0].data[0];
    const text = value.toFixed(1)+"%";

    ctx.font="bold 26px sans-serif";
    ctx.fillStyle="#22c55e";
    ctx.textAlign="center";
    ctx.textBaseline="middle";
    ctx.fillText(text,width/2,height/2);

    ctx.save();
  }
};

function App(){

  const [video,setVideo]=useState(null);
  const [playerName,setPlayerName]=useState("");
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);

  const safeArray=v=>Array.isArray(v)?v:[];

  const handleUpload=async()=>{
    if(!video) return alert("Upload video");

    setLoading(true);
    setResult(null);

    const fd=new FormData();
    fd.append("file",video);
    fd.append("player_name",playerName);

    try {
  const res = await axios.post(
    `${API_BASE_URL}/upload-video/`,
    fd
        );
      setResult(res.data);
    }catch(e){
      alert("Upload failed");
    }

    setLoading(false);
  };

  const radarData=result?{
    labels:["Technique","Speed","Control","Stamina","Agility"],
    datasets:[{
      label:"Athlete Profile",
      data:[
        result.avg_score,
        result.avg_score*0.9,
        result.avg_score*0.85,
        result.avg_score*0.8,
        result.avg_score*0.88
      ],
      backgroundColor:"rgba(34,197,94,0.3)",
      borderColor:"#22c55e"
    }]
  }:null;

  const donutData=result?{
    datasets:[{
      data:[result.avg_score,100-result.avg_score],
      backgroundColor:["#22c55e","#1f2937"]
    }]
  }:null;

  return(
<div style={{
  minHeight:"100vh",
  background:"linear-gradient(135deg,#020617,#0f172a,#020617)",
  color:"white",
  padding:"30px"
}}>

<h1 style={{
  textAlign:"center",
  fontSize:"34px",
  fontWeight:"bold",
  color:"white"
}}>

🏆

<span style={{
  background:"linear-gradient(90deg,#22c55e,#3b82f6)",
  WebkitBackgroundClip:"text",
  WebkitTextFillColor:"transparent",
  marginLeft:"10px"
}}>
Real-Time MultiSport Skill Estimator
</span>

</h1>



{/* INPUT PANEL */}
<div style={{
  width:"360px",
  margin:"40px auto",
  padding:"25px",
  borderRadius:"18px",
  backdropFilter:"blur(14px)",
  background:"rgba(255,255,255,0.05)",
  boxShadow:"0 0 30px rgba(0,255,150,0.15)"
}}>

<input
  placeholder="Player Name"
  value={playerName}
  onChange={e=>setPlayerName(e.target.value)}
  style={{
    width:"100%",
    padding:"12px",
    borderRadius:"10px",
    border:"1px solid #1f2937",
    background:"#0b1220",
    color:"white",
    marginBottom:"15px",
    outline:"none"
  }}
/>

<input
  type="file"
  accept="video/*"
  onChange={e=>setVideo(e.target.files[0])}
  style={{
    width:"100%",
    padding:"10px",
    borderRadius:"10px",
    background:"#0b1220",
    color:"white",
    border:"1px solid #1f2937",
    marginBottom:"20px"
  }}
/>

<button
  onClick={handleUpload}
  style={{
    width:"100%",
    padding:"14px",
    background:"linear-gradient(90deg,#22c55e,#3b82f6)",
    border:"none",
    borderRadius:"10px",
    color:"white",
    fontWeight:"bold",
    cursor:"pointer"
  }}
>
Analyze Performance
</button>

</div>



{/* LOADING */}
{loading && <div style={{textAlign:"center"}}>⏳ Analyzing...</div>}

{/* DASHBOARD */}
{result && (
<div style={{marginTop:"40px",display:"grid",gap:"25px",justifyContent:"center"}}>


{/* KPI */}
<div style={{display:"flex",gap:"20px",justifyContent:"center"}}>
{[
["Score",result.avg_score.toFixed(1)],
["Level",result.skill_level],
["Player",playerName||"N/A"]
].map((c,i)=>(
<div key={i} style={{
background:"#111827",padding:"20px",borderRadius:"14px",width:"150px",textAlign:"center"
}}>
<h3>{c[0]}</h3>
<h2 style={{color:"#22c55e"}}>{c[1]}</h2>
</div>
))}
</div>

{/* RADAR */}
<div style={{background:"#111827",padding:"25px",borderRadius:"18px",width:"420px"}}>
<h3>📡 Athlete Profile</h3>
<Radar data={radarData}/>
</div>

{/* DONUT */}
<div style={{background:"#111827",padding:"25px",borderRadius:"18px",width:"420px"}}>
<h3>🏅 Performance Score</h3>
<Doughnut
data={donutData}
plugins={[centerTextPlugin]}
options={{
cutout:"70%",
plugins:{legend:{display:false}}
}}
/>
</div>


{/* TIPS */}
<div style={{background:"#111827",padding:"20px",borderRadius:"16px",width:"420px"}}>
<h3>🎯 Improvement Plan</h3>
<ul>
{safeArray(result.tips).map((t,i)=>(<li key={i}>{t}</li>))}
</ul>
</div>

{/* WEAKNESS DETECTION */}
{result.weakness && result.weakness.length > 0 && (
<div style={{
  background:"#111827",
  padding:"20px",
  borderRadius:"16px",
  width:"420px"
}}>
<h3>⚠ Weakness Detection</h3>

<ul>
{result.weakness.map((w,i)=>(
<li key={i}>{w}</li>
))}
</ul>

</div>
)}

{/* HISTORY */}
{result?.history?.length > 0 && (
<div style={{background:"#111827",padding:"20px",width:"420px"}}>
<h3>📈 Performance History</h3>
<Line data={{
  labels: result.history.map(h=>h.date),
  datasets:[{
    label:"Score Trend",
    data: result.history.map(h=>h.score),
    borderColor:"#22c55e",
    tension:0.3
  }]
}}/>
</div>
)}

{result.improvement && (
<div className="card">
  <h3>📈 Progress Update</h3>
  <p>{result.improvement}</p>
</div>
)}

{result.prediction && (
<div className="card">
  <h3>🔮 AI Improvement Prediction</h3>
  <p style={{color:"#22c55e", fontWeight:"bold"}}>
    {result.prediction}
  </p>
</div>
)}

{result.medal && (
<div className="card">
  <h3>🏅 Achievement</h3>
  <h2>{result.medal}</h2>
</div>
)}

{result.leaderboard && (
<div className="card">
<h3>🏆 Leaderboard</h3>

{result.leaderboard.map((p,i)=>(
<p key={i}>
#{i+1} {p.name} — {p.score}
</p>
))}

</div>
)}



{/* PDF DOWNLOAD */}
{result.pdf_report && (
<div style={{
  textAlign:"center",
  marginTop:"10px"
}}>
<a
  href={`${API_BASE_URL}${result.pdf_report}`}
  target="_blank"
  rel="noreferrer"
>
<button style={{
  padding:"14px 28px",
  background:"linear-gradient(90deg,#22c55e,#3b82f6)",
  border:"none",
  borderRadius:"10px",
  color:"white",
  fontWeight:"bold",
  cursor:"pointer"
}}>
📄 Download PDF Report
</button>
</a>
</div>
)}


{/* VIDEO */}
{result.output_video && (
<div style={{background:"#111827",padding:"20px",borderRadius:"18px",width:"520px"}}>
<h3>🎥 Video Analysis</h3>
<video width="480" controls>
<source src={`${API_BASE_URL}${result.output_video}`} type="video/mp4"/>
</video>
</div>
)}

</div>
)}

</div>
);
}



export default App;
