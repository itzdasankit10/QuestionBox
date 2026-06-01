import { 
  MessageSquare, Loader, Download, BookOpen, RefreshCw, Users, 
  ArrowLeft, ArrowRight, ClipboardCheck, BarChart2, RotateCcw, 
  PlusCircle, CheckCircle, UploadCloud, FileText, File, AlertTriangle,
  User, Lock, Mail, BadgeInfo, LogIn, FileQuestion, ClipboardList
} from "lucide-react";
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Configuration
const API_URL = "http://localhost:8000";

const subjectOptions = {
  '6': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science', 'Science'],
  '7': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science', 'Science'],
  '8': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science', 'Science'],
  '9': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science', 'Science'],
  '10': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science', 'Science'],
};

const generatorModels = [
  "gpt2", "t5", "mistral", "gpt-4", "mistral_rank4", "gemma2:9b", "phi3-rank4", "llama3.1"
];
const metricOptions = [
  { value: "bert", label: "BERT (Semantic Similarity)" },
  { value: "bleu", label: "BLEU (Exact Match/Precision)" },
  { value: "rouge", label: "ROUGE (Recall)" },
  { value: "meteor", label: "METEOR (Harmonic Mean & Word Order)" }
];

// --- Sub-Component: Invigilator Rating Column ---
const InvigilatorRatingColumn = ({
  invigilatorId,
  title,
  quizData,
  ratings,
  onChange,
  onRemarkChange
}) => {
  const currentRatings = ratings[invigilatorId]?.ratings || {};
  const currentOverallRemarks = ratings[invigilatorId]?.overall_remarks || "";

  return (
    <div className="flex flex-col h-full min-h-0">
      <h4 className="text-lg font-bold text-sky-300 mb-3 text-center">{title}</h4>
      <div className='bg-slate-900/70 text-white rounded-xl p-6 shadow-inner border border-slate-700 overflow-y-auto text-left flex-grow min-h-0'>
        {(!quizData || Object.keys(quizData).length === 0) && (
          <p className="text-gray-400">No questions loaded.</p>
        )}
        {quizData && Object.keys(quizData).map(qType => (
          <div key={qType} className="mb-6">
            <h5 className="text-md font-bold text-sky-400 capitalize mb-3">{qType.replace(/_/g, ' ')}</h5>
            <ul className="list-decimal list-inside space-y-6">
              {quizData[qType]?.length === 0 && (
                <p className="text-gray-500 text-sm italic">No questions of this type.</p>
              )}
              {quizData[qType]?.map((item, i) => (
                <li key={i} className="space-y-3 pb-4 border-b border-slate-700/50 last:border-b-0">
                  <p className="whitespace-pre-line text-sm text-gray-200">{item.question}</p>
                  <div className="flex items-center gap-2">
                    <label htmlFor={`rating-${invigilatorId}-${qType}-${i}`} className="text-sm font-medium text-sky-300">Difficulty:</label>
                    <select
                      id={`rating-${invigilatorId}-${qType}-${i}`}
                      value={currentRatings[qType]?.[i]?.difficulty || ""}
                      onChange={(evt) => onChange(invigilatorId, qType, i, evt.target.value)}
                      className="text-xs p-1 rounded bg-slate-800 border border-slate-600 text-white focus:ring-1 focus:ring-sky-500 outline-none"
                    >
                      <option value="" disabled>Select...</option>
                      <option value="Easy">Easy</option>
                      <option value="Medium">Medium</option>
                      <option value="Hard">Hard</option>
                      <option value="Irrelevant">Irrelevant</option>
                    </select>
                  </div>
                  <div className="mt-2">
                    <label htmlFor={`remarks-${invigilatorId}-${qType}-${i}`} className="block text-sm font-medium text-sky-300 mb-1">Remarks:</label>
                    <textarea
                      id={`remarks-${invigilatorId}-${qType}-${i}`}
                      placeholder="Add remarks..."
                      className="w-full p-2 text-sm rounded bg-slate-800 border border-slate-600 text-white focus:ring-1 focus:ring-sky-500 outline-none"
                      rows="3"
                      value={currentRatings[qType]?.[i]?.remarks || ""}
                      onChange={(e) => onRemarkChange(invigilatorId, qType, i, e.target.value)}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
        <div className="mt-6">
          <label htmlFor={`overall-remarks-${invigilatorId}`} className="block text-sm font-medium text-sky-300 mb-2">
            Overall Remarks:
          </label>
          <textarea
            id={`overall-remarks-${invigilatorId}`}
            placeholder="Add overall remarks..."
            className="w-full p-2 text-sm rounded bg-slate-800 border border-slate-600 text-white focus:ring-1 focus:ring-sky-500 outline-none"
            rows="4"
            value={currentOverallRemarks}
            onChange={(e) => onRemarkChange(invigilatorId, null, null, e.target.value)}
          />
        </div>
      </div>
    </div>
  );
};

// --- Main App Component ---
export default function App() {
  // Mode Selection: null=initial, 'generate'=Question Generation, 'evaluate'=Question Evaluation
  const [appMode, setAppMode] = useState(null);
  
  // Steps for Generation Mode: 1=Auth, 2=GuestEmail, 3=Context, 4=Result
  // Steps for Evaluation Mode: 1=Auth, 2=GuestEmail, 3=LoadQuestions, 4=HumanEval, 5=MetricEval, 6=Rating
  const [step, setStep] = useState(1);
  
  const [classNum, setClassNum] = useState("");
  const [subject, setSubject] = useState("");
  const [quizData, setQuizData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [questionCounts, setQuestionCounts] = useState({ mcqs: 0, fill_in_the_blanks: 0, subjective: 0 });
  const [contextText, setContextText] = useState("");
  const [invigilatorRatings, setInvigilatorRatings] = useState({});
  const [currentInvigilatorId, setCurrentInvigilatorId] = useState(1);
  const [finalResults, setFinalResults] = useState(null);
  const [selectedModel, setSelectedModel] = useState(generatorModels[0]); 
  const [selectedMetric, setSelectedMetric] = useState("bert");
  
  // Auth State
  const [isSignup, setIsSignup] = useState(false);
  const [authData, setAuthData] = useState({ name: "", rollNo: "", designation: "", password: "" });
  const [guestEmail, setGuestEmail] = useState("");
  const [emailError, setEmailError] = useState("");
  
  // API Availability
  const [openaiAvailable, setOpenaiAvailable] = useState(false);

  // File Upload State
  const [uploadMode, setUploadMode] = useState(false);
  const [uploadedContext, setUploadedContext] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState("");

  // Evaluation Mode States
  const [savedQuestionPapers, setSavedQuestionPapers] = useState([]);
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [humanEvaluations, setHumanEvaluations] = useState({});
  const [referenceFile, setReferenceFile] = useState(null);

  // Check Backend on Mount
  useEffect(() => {
    fetch(`${API_URL}/check-models`)
      .then(res => res.json())
      .then(data => {
        const isReady = data.openai_client && data.openai_client.includes("Initialized");
        setOpenaiAvailable(isReady);
      })
      .catch(err => {
        console.error("Backend check failed:", err);
        setOpenaiAvailable(false); 
      });

    // Load saved question papers from localStorage
      // Load papers from backend GENERATED_PAPERS folder
    fetch(`${API_URL}/generated-papers`)
      .then(res => res.json())
      .then(data => {
        setSavedQuestionPapers(data);
      })
      .catch(err => {
        console.error("Failed loading papers:", err);
      });
    }, []);

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => setStep((prev) => prev - 1);

  const resetToModeSelection = () => {
    setAppMode(null);
    setStep(1);
    setClassNum("");
    setSubject("");
    setQuizData(null);
    setError("");
    setContextText("");
    setInvigilatorRatings({});
    setCurrentInvigilatorId(1);
    setFinalResults(null);
    setUploadMode(false);
    setUploadedContext("");
    setUploadedFileName("");
    setSelectedPaperId("");
    setHumanEvaluations({});
    setReferenceFile(null);
  };

  // --- AUTH HANDLERS ---
  const handleAuthChange = (e) => {
    setAuthData({ ...authData, [e.target.name]: e.target.value });
  };

  const handleAuthSubmit = () => {
    if (isSignup) {
       if(!authData.name || !authData.rollNo || !authData.designation || !authData.password) {
         alert("Please fill all fields"); return;
       }
    } else {
       if(!authData.rollNo || !authData.password) {
         alert("Please fill all fields"); return;
       }
    }
    setStep(3); // Jump to main content
  };

  const handleGuestClick = () => {
    setStep(2); // Go to Guest Email Step
  };

  const validateEmail = (email) => {
    return String(email)
      .toLowerCase()
      .match(
        /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|.(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      );
  };

  const handleGuestSubmit = () => {
    if (validateEmail(guestEmail)) {
      setEmailError("");
      setStep(3);
    } else {
      setEmailError("Please enter a valid email address.");
    }
  };

  const logout = () => {
    setStep(1);
    setAuthData({ name: "", rollNo: "", designation: "", password: "" });
    setGuestEmail("");
  };

  // --- GENERATION MODE HANDLERS ---
  const createApiFormData = (includeModel = false) => {
    const formData = new FormData();
    if (uploadMode) {
       formData.append('custom_context', uploadedContext);
    } else {
       if (!classNum || !subject) return null;
       formData.append('class_num', classNum);
       formData.append('subject', subject);
    }
    if (includeModel) {
      formData.append('model_name', selectedModel);
    }
    return formData;
  };

  const handleClassChange = (evt) => { setClassNum(evt.target.value); setSubject(""); };
  const handleSubjectChange = (evt) => { setSubject(evt.target.value); };
  const handleCountChange = (evt) => {
    const { name, value } = evt.target;
    const count = Math.max(0, parseInt(value, 10) || 0);
    setQuestionCounts(prev => ({ ...prev, [name]: count }));
  };
  const handleModelChange = (evt) => { setSelectedModel(evt.target.value); };

  const handleFileUpload = async (evt) => {
    const file = evt.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setError("");
    setUploadedContext("");
    setUploadedFileName(file.name);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/extract-text`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Text extraction failed.");
      }
      const data = await response.json();
      setUploadedContext(data.extracted_text);
    } catch (err) {
      setError("OCR/Extraction Error: " + err.message);
      setUploadedFileName("");
    } finally {
      setIsUploading(false);
    }
  };

  const generateQuestions = async () => {
    if (uploadMode) {
      if (!uploadedContext) { setError("Please upload a document first."); return; }
    } else {
      if (!classNum || !subject) { setError("Please select Class and Subject."); return; }
    }

    setLoading(true);
    setError("");
    setQuizData(null);
    setContextText("");

    const formData = createApiFormData(true);
    if (!formData) { setError("Input Missing."); setLoading(false); return; }
    
    formData.append('counts_json', JSON.stringify(questionCounts));
    
    try {
      const response = await fetch(`${API_URL}/generate-questions`, { method: "POST", body: formData });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to generate questions.");
      }
      const data = await response.json();
      setQuizData(data.questions);
      setContextText(data.context);
      nextStep();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadQuestionPaper = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/download-questions-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answered_questions: quizData, model_name: selectedModel }),
      });
      if (!response.ok) throw new Error('Question Paper PDF generation failed.');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `question_paper_${selectedModel}_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const saveQuestionPaper = () => {
    const paperId = `paper_${Date.now()}`;
    const paperData = {
      id: paperId,
      timestamp: new Date().toISOString(),
      classNum: uploadMode ? "Custom" : classNum,
      subject: uploadMode ? "Uploaded" : subject,
      model: selectedModel,
      questions: quizData,
      context: contextText
    };

    const updated = [...savedQuestionPapers, paperData];
    setSavedQuestionPapers(updated);
    // localStorage.setItem('savedQuestionPapers', JSON.stringify(updated));
    alert('Question paper saved successfully! You can now evaluate it.');
  };

  // --- EVALUATION MODE HANDLERS ---
  const loadQuestionPaper = () => {
    if (!selectedPaperId) {
      setError("Please select a question paper.");
      return;
    }

    const paper = savedQuestionPapers.find(
      p => (p.id || p.filename) === selectedPaperId
    );
    if (!paper) {
      setError("Question paper not found.");
      return;
    }

    setQuizData(paper.questions);
    setContextText(paper.context);
    setClassNum(paper.classNum);
    setSubject(paper.subject);
    setSelectedModel(paper.model);
    
    // Initialize human evaluations
    const initialEvals = {};
    Object.keys(paper.questions).forEach(qType => {
      initialEvals[qType] = {};
      paper.questions[qType].forEach((_, idx) => {
        initialEvals[qType][idx] = "";
      });
    });
    setHumanEvaluations(initialEvals);
    
    nextStep(); // Go to human evaluation step
  };

  const handleHumanEvaluationChange = (qType, qIndex, value) => {
    setHumanEvaluations(prev => ({
      ...prev,
      [qType]: {
        ...prev[qType],
        [qIndex]: value
      }
    }));
  };

  const submitHumanEvaluations = () => {
    // Check if all questions have been evaluated
    let allEvaluated = true;
    Object.keys(quizData).forEach(qType => {
      quizData[qType].forEach((_, idx) => {
        if (!humanEvaluations[qType]?.[idx]) {
          allEvaluated = false;
        }
      });
    });

    if (!allEvaluated) {
      if (!window.confirm("Some questions haven't been evaluated. Continue anyway?")) {
        return;
      }
    }

    nextStep(); // Go to metric evaluation
  };

  const handleReferenceFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setReferenceFile(file);
    }
  };

  const handleMetricEvaluate = async (qType, qIndex) => {
    const updatedData = JSON.parse(JSON.stringify(quizData));
    const item = updatedData[qType][qIndex];
    
    let cleanQ = item.question;
    if (cleanQ.includes("**Options:**")) { cleanQ = cleanQ.split("**Options:**")[0]; }
    else if (cleanQ.includes("\nA)")) { cleanQ = cleanQ.split("\nA)")[0]; }
    
    const prefixes = [/^Here.*?text:/i, /^Here is a new question:/i, /^\*\*Question:\*\*/i, /^\d+\.\s*/];
    prefixes.forEach(prefix => { cleanQ = cleanQ.replace(prefix, ""); });
    cleanQ = cleanQ.replace(/\*\*/g, "").trim();

    item.evaluating = true;
    setQuizData(updatedData);

    const formData = new FormData();
    
    // Pass class and subject for reference question matching
    if (classNum && classNum !== "Custom") {
      formData.append('class_num', classNum);
      formData.append('subject', subject);
    } else if (uploadMode || classNum === "Custom") {
      formData.append('custom_context', contextText);
    }
    
    // Append Reference File if it exists
    if (referenceFile) {
      formData.append('reference_file', referenceFile);
    }

    formData.append('question', cleanQ);
    formData.append('generated_answer', humanEvaluations[qType]?.[qIndex] || "No answer provided");
    formData.append('q_type', qType);
    formData.append('metric', selectedMetric);

    try {
      const response = await fetch(`${API_URL}/evaluate-accuracy`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Evaluation failed.');
      const result = await response.json();
      
      item.score = result.similarity_score;
      item.bert_answer = result.bert_answer;
      item.human_answer = humanEvaluations[qType]?.[qIndex];
    } catch (err) {
      item.score = 'Error';
      console.error("Evaluation error:", err);
    } finally {
      item.evaluating = false;
      setQuizData(JSON.parse(JSON.stringify(updatedData)));
    }
  };

  const goToRatingPage = () => {
    setStep(6); // Go to rating page
    setCurrentInvigilatorId(1);
  };

  const handleInvigilatorBack = () => {
    if (currentInvigilatorId > 1) {
      setCurrentInvigilatorId(prev => prev - 1);
    } else {
      setStep(5); // Back to metric evaluation
    }
  };

  const handleAddInvigilator = () => {
    setCurrentInvigilatorId(prev => prev + 1);
  };

  const ensureRatingPath = (ratings, invigilatorId, qType, qIndex) => {
    if (!ratings[invigilatorId]) ratings[invigilatorId] = { ratings: {}, overall_remarks: "" };
    if (!ratings[invigilatorId].ratings) ratings[invigilatorId].ratings = {};
    if (!ratings[invigilatorId].ratings[qType]) ratings[invigilatorId].ratings[qType] = {};
    if (!ratings[invigilatorId].ratings[qType][qIndex]) 
      ratings[invigilatorId].ratings[qType][qIndex] = { difficulty: "", remarks: "" };
  };

  const handleRatingChange = (invigilatorId, qType, i, value) => {
    setInvigilatorRatings(prev => {
      const updated = JSON.parse(JSON.stringify(prev));
      ensureRatingPath(updated, invigilatorId, qType, i);
      updated[invigilatorId].ratings[qType][i].difficulty = value;
      return updated;
    });
  };
    
  const handleRemarkChange = (invigilatorId, qType, i, value) => {
    setInvigilatorRatings(prev => {
      const updated = JSON.parse(JSON.stringify(prev));
      if (qType === null) {
        if (!updated[invigilatorId]) updated[invigilatorId] = { ratings: {}, overall_remarks: "" };
        updated[invigilatorId].overall_remarks = value;
      } else {
        ensureRatingPath(updated, invigilatorId, qType, i);
        updated[invigilatorId].ratings[qType][i].remarks = value;
      }
      return updated;
    });
  };

  const handleSubmitRatings = async () => {
    let combinedOverall = "";
    const collatedQuestionRemarks = {};
    
    const invigilatorKeys = Object.keys(invigilatorRatings).filter(key => key.startsWith('invigilator')).sort();

    if (invigilatorKeys.length === 0) {
      if(!window.confirm("No ratings have been entered. Do you want to submit anyway?")) return;
    }

    invigilatorKeys.forEach(key => {
      const num = key.replace('invigilator', '');
      const remark = invigilatorRatings[key]?.overall_remarks;
      if (remark) {
        combinedOverall += `Expert ${num}:\n${remark}\n\n`;
      }
    });

    if (quizData) {
      Object.keys(quizData).forEach(qType => {
        collatedQuestionRemarks[qType] = {};
        quizData[qType].forEach((item, i) => {
          const remarksForThisQ = [];
          
          invigilatorKeys.forEach(key => {
             const num = key.replace('invigilator', '');
             const diff = invigilatorRatings[key]?.ratings?.[qType]?.[i]?.difficulty;
             const rem = invigilatorRatings[key]?.ratings?.[qType]?.[i]?.remarks;
             
             if (diff || rem) {
               remarksForThisQ.push(`Inv ${num}: [${diff || 'N/A'}] ${rem || ''}`);
             }
          });

          if (remarksForThisQ.length > 0) {
            collatedQuestionRemarks[qType][i] = remarksForThisQ;
          }
        });
      });
    }

    setFinalResults({
      overallRemarks: combinedOverall.trim() || "No overall remarks submitted.",
      questionRemarks: collatedQuestionRemarks
    });

    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/submit-difficulty-ratings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          class_num: classNum,
          subject: subject,
          invigilator_data: invigilatorRatings,
        }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to submit ratings.');
      }
      await response.json();
      alert('Ratings submitted successfully!');
      setStep(5); // Back to metric evaluation to show final results
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const ScoreBadge = ({ score }) => {
    if (score === 'Error' || typeof score !== 'number') return <span className="text-red-500">Error</span>;
    const numScore = (Math.min(score, 1) * 100).toFixed(1);
    let color = 'text-gray-400';
    if (score > 0.8) color = 'text-green-400';
    else if (score > 0.6) color = 'text-yellow-400';
    else if (score < 0.3) color = 'text-red-400';
    else if (score < 0) color = 'text-red-600 font-bold';
    if (isNaN(score)) return <span className="text-orange-500">Invalid Score</span>;
    return <span className={`${color} font-bold`}>Score: {numScore}%</span>;
  };

  const isOpenAIModel = selectedModel.includes("gpt-3") || selectedModel.includes("gpt-4");
  const isApiKeyMissing = isOpenAIModel && !openaiAvailable;

  return (
    <div className='flex flex-col items-center min-h-full w-full bg-gradient-to-br from-slate-900 to-gray-900 text-gray-200 p-6'>
      <AnimatePresence mode='wait'>
        
        {/* --- MODE SELECTION SCREEN --- */}
        {appMode === null && (
          <motion.div
            key='mode-selection'
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-2xl shadow-2xl border border-slate-700 mt-20'
          >
            <div className="flex flex-col items-center mb-8">
              <div className="p-4 bg-sky-500/20 rounded-full mb-4">
                <BookOpen className="w-12 h-12 text-sky-400" />
              </div>
              <h2 className='text-3xl font-bold text-sky-300 mb-2'>
                Assessment System
              </h2>
              <p className="text-sm text-gray-400 text-center">
                Choose your operation mode
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => { setAppMode('generate'); setStep(1); }}
                className="group relative overflow-hidden bg-gradient-to-br from-sky-500/20 to-blue-600/20 border-2 border-sky-500/50 rounded-xl p-8 hover:border-sky-400 transition-all"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-sky-500/0 to-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative z-10 flex flex-col items-center">
                  <FileQuestion className="w-16 h-16 text-sky-400 mb-4" />
                  <h3 className="text-xl font-bold text-white mb-2">Question Generation</h3>
                  <p className="text-sm text-gray-400 text-center">
                    Generate new question papers using AI models
                  </p>
                </div>
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => { setAppMode('evaluate'); setStep(1); }}
                className="group relative overflow-hidden bg-gradient-to-br from-green-500/20 to-emerald-600/20 border-2 border-green-500/50 rounded-xl p-8 hover:border-green-400 transition-all"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-green-500/0 to-emerald-600/10 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative z-10 flex flex-col items-center">
                  <ClipboardList className="w-16 h-16 text-green-400 mb-4" />
                  <h3 className="text-xl font-bold text-white mb-2">Question Evaluation</h3>
                  <p className="text-sm text-gray-400 text-center">
                    Evaluate saved question papers with human & AI metrics
                  </p>
                </div>
              </motion.button>
            </div>
          </motion.div>
        )}

        {/* --- STEP 1: LOGIN / SIGNUP (Both Modes) --- */}
        {appMode !== null && step === 1 && (
          <motion.div
            key='step1'
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-md shadow-2xl border border-slate-700 mt-20'
          >
            <button
              onClick={resetToModeSelection}
              className="absolute top-4 left-4 text-gray-500 hover:text-white"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>

            <div className="flex flex-col items-center mb-6">
               <div className="p-4 bg-sky-500/20 rounded-full mb-4">
                  <LogIn className="w-8 h-8 text-sky-400" />
               </div>
               <h2 className='text-2xl font-bold text-sky-300'>
                  {isSignup ? "Faculty Registration" : "Portal Login"}
               </h2>
               <p className="text-sm text-gray-400 mt-1">
                  {appMode === 'generate' ? 'Question Generation Mode' : 'Question Evaluation Mode'}
               </p>
            </div>

            <div className="space-y-4">
               {isSignup && (
                 <>
                   <div className="relative">
                      <User className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
                      <input 
                        name="name" placeholder="Full Name" 
                        value={authData.name} onChange={handleAuthChange}
                        className="w-full pl-10 p-3 rounded-lg bg-slate-900/50 border border-slate-700 text-white focus:ring-2 focus:ring-sky-500 outline-none"
                      />
                   </div>
                   <div className="relative">
                      <BadgeInfo className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
                      <input 
                        name="designation" placeholder="Designation" 
                        value={authData.designation} onChange={handleAuthChange}
                        className="w-full pl-10 p-3 rounded-lg bg-slate-900/50 border border-slate-700 text-white focus:ring-2 focus:ring-sky-500 outline-none"
                      />
                   </div>
                 </>
               )}
               
               <div className="relative">
                  <FileText className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
                  <input 
                    name="rollNo" placeholder="Roll Number / ID" 
                    value={authData.rollNo} onChange={handleAuthChange}
                    className="w-full pl-10 p-3 rounded-lg bg-slate-900/50 border border-slate-700 text-white focus:ring-2 focus:ring-sky-500 outline-none"
                  />
               </div>

               <div className="relative">
                  <Lock className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
                  <input 
                    name="password" type="password" placeholder="Password" 
                    value={authData.password} onChange={handleAuthChange}
                    className="w-full pl-10 p-3 rounded-lg bg-slate-900/50 border border-slate-700 text-white focus:ring-2 focus:ring-sky-500 outline-none"
                  />
               </div>
            </div>

            <button 
               onClick={handleAuthSubmit}
               className="w-full mt-6 py-3 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-xl shadow-lg hover:shadow-sky-500/20 hover:opacity-90 font-bold transition"
            >
               {isSignup ? "Create Account" : "Log In"}
            </button>

            <div className="mt-4 flex flex-col items-center gap-2">
               <button onClick={() => setIsSignup(!isSignup)} className="text-xs text-sky-400 hover:text-sky-300">
                  {isSignup ? "Already have an account? Log In" : "New User? Create Account"}
               </button>
               
               <div className="w-full border-t border-slate-700 my-2"></div>
               
               <button onClick={handleGuestClick} className="text-sm text-gray-400 hover:text-white transition flex items-center gap-2">
                  <Users className="w-4 h-4" /> Continue as Guest
               </button>
            </div>
          </motion.div>
        )}

        {/* --- STEP 2: GUEST EMAIL (Both Modes) --- */}
        {appMode !== null && step === 2 && (
           <motion.div
             key='step2'
             initial={{ opacity: 0, x: 50 }}
             animate={{ opacity: 1, x: 0 }}
             exit={{ opacity: 0, x: -50 }}
             className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-md shadow-2xl border border-slate-700 mt-20'
           >
              <button onClick={() => setStep(1)} className="absolute top-4 left-4 text-gray-500 hover:text-white">
                 <ArrowLeft className="w-5 h-5" />
              </button>
              
              <div className="flex flex-col items-center mb-6">
                 <Mail className="w-12 h-12 text-green-400 mb-4" />
                 <h2 className='text-xl font-bold text-gray-200'>Guest Access</h2>
                 <p className="text-sm text-gray-400 text-center">Please enter a valid email to continue.</p>
              </div>

              <div className="space-y-4">
                 <input 
                   placeholder="Enter your email" 
                   value={guestEmail} onChange={(e) => setGuestEmail(e.target.value)}
                   className={`w-full p-3 rounded-lg bg-slate-900/50 border ${emailError ? 'border-red-500' : 'border-slate-700'} text-white focus:ring-2 focus:ring-green-500 outline-none`}
                 />
                 {emailError && <p className="text-xs text-red-400">{emailError}</p>}
              </div>

              <button 
                 onClick={handleGuestSubmit}
                 className="w-full mt-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl shadow-lg hover:opacity-90 font-bold transition"
              >
                 Verify & Continue
              </button>
           </motion.div>
        )}

        {/* ========== GENERATION MODE SCREENS ========== */}
        
        {/* --- STEP 3: CONTEXT & CONFIG (Generation Mode) --- */}
        {appMode === 'generate' && step === 3 && (
          <motion.div
            key='generate-step3'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-3xl shadow-2xl border border-slate-700'
          >
            <div className="flex justify-between items-center mb-6">
               <h2 className='text-2xl font-bold text-sky-300 flex items-center'>
                 <BookOpen className='w-6 h-6 mr-2 text-sky-400' />
                 Configure Question Generation
               </h2>
               <button onClick={logout} className="text-xs text-red-400 hover:text-red-300 border border-red-900/50 bg-red-900/20 px-3 py-1 rounded">
                  Log Out
               </button>
            </div>

            <div className="flex space-x-6 mb-8 border-b border-slate-700">
               <button 
                  onClick={() => setUploadMode(false)}
                  className={`pb-3 text-sm font-bold transition duration-300 ${!uploadMode ? 'text-sky-400 border-b-2 border-sky-400' : 'text-gray-400 hover:text-gray-200'}`}
               >
                  Select from Books
               </button>
               <button 
                  onClick={() => setUploadMode(true)}
                  className={`pb-3 text-sm font-bold transition duration-300 ${uploadMode ? 'text-sky-400 border-b-2 border-sky-400' : 'text-gray-400 hover:text-gray-200'}`}
               >
                  Upload / OCR (Lens)
               </button>
            </div>

            {!uploadMode && (
              <div className='w-full grid grid-cols-1 md:grid-cols-2 gap-6 mb-6'>
                <div>
                  <label htmlFor="class-select" className="block text-sm font-medium text-sky-300 mb-2">Class</label>
                  <select id="class-select" value={classNum} onChange={handleClassChange} className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none">
                    <option value="" disabled>Select a class</option>
                    {Object.keys(subjectOptions).map(cls => <option key={cls} value={cls}>Class {cls}</option>)}
                  </select>
                </div>
                <div>
                  <label htmlFor="subject-select" className="block text-sm font-medium text-sky-300 mb-2">Subject</label>
                  <select id="subject-select" value={subject} onChange={handleSubjectChange} disabled={!classNum} className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none disabled:opacity-50">
                    <option value="" disabled>Select a subject</option>
                    {classNum && subjectOptions[classNum].map(sub => <option key={sub} value={sub}>{sub}</option>)}
                  </select>
                </div>
              </div>
            )}

            {uploadMode && (
               <div className="mb-6">
                  <div className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-8 transition duration-300 ${uploadedContext ? 'border-green-500/50 bg-green-900/10' : 'border-slate-600 bg-slate-900/40 hover:border-sky-500/50'}`}>
                     <input 
                        type="file" 
                        id="file-upload" 
                        className="hidden" 
                        accept=".pdf,.txt,.png,.jpg,.jpeg,.webp"
                        onChange={handleFileUpload}
                     />
                     
                     {!uploadedContext && !isUploading && (
                        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center group w-full h-full">
                           <div className="p-4 bg-slate-800 rounded-full mb-3 group-hover:bg-slate-700 transition transform group-hover:scale-110">
                              <UploadCloud className="w-10 h-10 text-sky-400" />
                           </div>
                           <span className="text-sky-300 font-medium text-lg">Click to Upload Document</span>
                           <span className="text-xs text-gray-500 mt-2">Supports PDF, TXT, Images (OCR)</span>
                        </label>
                     )}

                     {isUploading && (
                        <div className="flex flex-col items-center">
                           <Loader className="animate-spin w-12 h-12 text-sky-400 mb-4" />
                           <span className="text-gray-300 font-medium text-lg">Extracting Text...</span>
                        </div>
                     )}

                     {uploadedContext && !isUploading && (
                        <div className="w-full flex flex-col items-center">
                           <div className="flex items-center text-green-400 mb-4 gap-2 bg-green-900/20 px-4 py-2 rounded-full border border-green-500/30">
                              <CheckCircle className="w-5 h-5" />
                              <span className="font-bold">{uploadedFileName} Processed</span>
                           </div>
                           <div className="w-full bg-slate-950/60 rounded-lg p-4 h-32 overflow-y-auto text-left border border-slate-800 font-mono text-xs text-gray-400">
                              {uploadedContext.substring(0, 500)}...
                           </div>
                           <label htmlFor="file-upload" className="mt-4 text-sm text-sky-400 cursor-pointer hover:text-sky-300 flex items-center gap-2">
                              <RefreshCw className="w-3 h-3" /> Upload Different File
                           </label>
                        </div>
                     )}
                  </div>
               </div>
            )}

            <div className="mb-6">
              <label htmlFor="model-select" className="block text-sm font-medium text-sky-300 mb-2">Select Generator Model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={handleModelChange}
                className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none"
              >
                {generatorModels.map(modelName => (
                  <option key={modelName} value={modelName}>{modelName}</option>
                ))}
              </select>
            </div>

            <div className="w-full grid grid-cols-3 gap-4 my-6">
              <div>
                <label htmlFor="mcqs" className="block text-xs font-bold text-sky-300 mb-2 uppercase">MCQs</label>
                <input type="number" name="mcqs" id="mcqs" value={questionCounts.mcqs} onChange={handleCountChange} min="0" className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none text-center" />
              </div>
              <div>
                <label htmlFor="fill_in_the_blanks" className="block text-xs font-bold text-sky-300 mb-2 uppercase">Fill Blanks</label>
                <input type="number" name="fill_in_the_blanks" id="fill_in_the_blanks" value={questionCounts.fill_in_the_blanks} onChange={handleCountChange} min="0" className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none text-center" />
              </div>
              <div>
                <label htmlFor="subjective" className="block text-xs font-bold text-sky-300 mb-2 uppercase">Subjective</label>
                <input type="number" name="subjective" id="subjective" value={questionCounts.subjective} onChange={handleCountChange} min="0" className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none text-center" />
              </div>
            </div>
            
            {error && <p className="text-red-400 mt-4 text-center bg-red-900/20 p-2 rounded border border-red-500/30 text-sm">{error}</p>}
            
            {isApiKeyMissing && (
                <div className="mt-4 p-3 bg-red-900/20 border border-red-500/50 rounded-lg flex items-center justify-center gap-2 text-red-300">
                    <AlertTriangle className="w-5 h-5" />
                    <span className="font-bold text-sm">⚠️ OpenAI API Key not found on server.</span>
                </div>
            )}

            <div className='flex justify-between mt-8'>
              <button 
                onClick={resetToModeSelection}
                className="px-6 py-3 bg-slate-700 text-gray-200 rounded-xl hover:bg-slate-600 transition"
              >
                ← Back to Menu
              </button>
              <button 
                onClick={generateQuestions} 
                disabled={loading || (uploadMode && !uploadedContext) || (!uploadMode && (!classNum || !subject)) || isApiKeyMissing}
                className={`px-8 py-3 rounded-xl shadow-lg transition flex items-center font-bold text-lg
                    ${isApiKeyMissing 
                        ? 'bg-slate-700 text-gray-500 cursor-not-allowed' 
                        : 'bg-gradient-to-r from-sky-500 to-blue-600 text-white hover:shadow-sky-500/20 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed'
                    }
                `}
              >
                {loading && <Loader className="animate-spin w-5 h-5 mr-3" />}
                {loading ? "Generating..." : "Generate Questions →"}
              </button>
            </div>
          </motion.div>
        )}

        {/* --- STEP 4: GENERATED QUESTIONS (Generation Mode) --- */}
        {appMode === 'generate' && step === 4 && (
          <motion.div
            key='generate-step4'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-4xl shadow-2xl border border-slate-700'
          >
            <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-6">
                <h2 className='text-2xl font-bold text-sky-300 flex items-center'>
                <MessageSquare className='w-6 h-6 mr-2 text-sky-400' />
                Generated Questions
                </h2>
                <div className="flex items-center gap-4 mt-2 md:mt-0">
                  <span className="text-xs bg-slate-700 px-2 py-1 rounded text-gray-300">Model: {selectedModel}</span>
                  {uploadMode ? (
                      <span className="text-xs bg-green-900/40 text-green-300 px-2 py-1 rounded border border-green-700/50 flex items-center gap-1">
                        <FileText className="w-3 h-3" /> Custom Doc
                      </span>
                  ) : (
                      <span className="text-xs bg-blue-900/40 text-blue-300 px-2 py-1 rounded border border-blue-700/50 flex items-center gap-1">
                        <BookOpen className="w-3 h-3" /> Class {classNum} - {subject}
                      </span>
                  )}
                </div>
            </div>

            {error && <p className="text-red-400 mb-4">{error}</p>}
            
            <div className='bg-slate-900/70 text-white rounded-xl p-6 shadow-inner border border-slate-700 h-[500px] overflow-y-auto text-left'>
              {quizData && Object.keys(quizData).map(qType => (
                <div key={qType} className="mb-6">
                  <h3 className="text-xl font-bold text-sky-400 capitalize mb-3">{qType.replace(/_/g, ' ')}</h3>
                  <ul className="list-decimal list-inside space-y-4">
                    {quizData[qType]?.length === 0 && (
                      <p className="text-gray-500 text-sm italic">No questions generated.</p>
                    )}
                    {quizData[qType]?.map((item, i) => (
                      <li key={i} className="pb-4 border-b border-slate-700/50 last:border-b-0">
                        <p className="whitespace-pre-line text-gray-200">{item.question}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className='flex justify-between items-center mt-6 w-full gap-2'>
              <button 
                onClick={() => setStep(3)} 
                className='px-4 py-2 bg-slate-700 text-gray-200 rounded-lg hover:bg-slate-600 transition'
              >
                ← Back
              </button>
              
              <div className="flex gap-2">
                  <button 
                      onClick={saveQuestionPaper} 
                      disabled={loading || !quizData} 
                      className='px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition flex items-center'
                  >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Save for Evaluation
                  </button>

                  <button 
                      onClick={downloadQuestionPaper} 
                      disabled={loading || !quizData} 
                      className='px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition flex items-center'
                  >
                      {loading ? <Loader className="animate-spin w-4 h-4 mr-2" /> : <Download className="w-4 h-4 mr-2" />}
                      Download PDF
                  </button>

                  <button 
                      onClick={resetToModeSelection}
                      className='px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg shadow hover:opacity-90 transition'
                  >
                      🔄 New Session
                  </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* ========== EVALUATION MODE SCREENS ========== */}

        {/* --- STEP 3: LOAD SAVED QUESTIONS (Evaluation Mode) --- */}
        {appMode === 'evaluate' && step === 3 && (
          <motion.div
            key='evaluate-step3'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-3xl shadow-2xl border border-slate-700'
          >
            <div className="flex justify-between items-center mb-6">
               <h2 className='text-2xl font-bold text-sky-300 flex items-center'>
                 <ClipboardList className='w-6 h-6 mr-2 text-sky-400' />
                 Load Question Paper
               </h2>
               <button onClick={logout} className="text-xs text-red-400 hover:text-red-300 border border-red-900/50 bg-red-900/20 px-3 py-1 rounded">
                  Log Out
               </button>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-sky-300 mb-2">
                Select a saved question paper to evaluate:
              </label>
              
              {savedQuestionPapers.length === 0 ? (
                <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-8 text-center">
                  <FileQuestion className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400 mb-2">No saved question papers found</p>
                  <p className="text-sm text-gray-500">Generate questions first to evaluate them</p>
                </div>
              ) : (
                <select
                  value={selectedPaperId}
                  onChange={(e) => setSelectedPaperId(e.target.value)}
                  className="w-full p-3 rounded-lg bg-slate-900/70 text-white border border-slate-700 focus:ring-2 focus:ring-sky-500 outline-none"
                >
                  <option value="" disabled>Choose a question paper...</option>
                  {savedQuestionPapers.map(paper => (
                    <option key={paper.id} value={paper.id}>
                      {new Date(paper.timestamp).toLocaleString()} - {paper.classNum} {paper.subject} ({paper.model})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {error && <p className="text-red-400 mt-4 text-center bg-red-900/20 p-2 rounded border border-red-500/30 text-sm">{error}</p>}

            <div className='flex justify-between mt-8'>
              <button 
                onClick={resetToModeSelection}
                className="px-6 py-3 bg-slate-700 text-gray-200 rounded-xl hover:bg-slate-600 transition"
              >
                ← Back to Menu
              </button>
              <button 
                onClick={loadQuestionPaper}
                disabled={!selectedPaperId || savedQuestionPapers.length === 0}
                className="px-8 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl shadow-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition font-bold"
              >
                Load & Start Evaluation →
              </button>
            </div>
          </motion.div>
        )}

        {/* --- STEP 4: HUMAN EVALUATION (Evaluation Mode) --- */}
        {appMode === 'evaluate' && step === 4 && (
          <motion.div
            key='evaluate-step4'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-4xl shadow-2xl border border-slate-700'
          >
            <h2 className='text-2xl font-bold text-sky-300 mb-6 flex items-center'>
              <User className='w-6 h-6 mr-2 text-sky-400' />
              Human Evaluation
            </h2>

            <div className="mb-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
              <p className="text-sm text-blue-200">
                <strong>Instructions:</strong> Provide your answer for each question. This will be used as the reference for comparison in the metric evaluation.
              </p>
            </div>

            <div className='bg-slate-900/70 text-white rounded-xl p-6 shadow-inner border border-slate-700 h-[500px] overflow-y-auto text-left'>
              {quizData && Object.keys(quizData).map(qType => (
                <div key={qType} className="mb-6">
                  <h3 className="text-xl font-bold text-sky-400 capitalize mb-3">{qType.replace(/_/g, ' ')}</h3>
                  <ul className="list-decimal list-inside space-y-6">
                    {quizData[qType]?.map((item, i) => (
                      <li key={i} className="space-y-3 pb-4 border-b border-slate-700/50 last:border-b-0">
                        <p className="whitespace-pre-line text-gray-200">{item.question}</p>
                        <div className="mt-2">
                          <label className="block text-sm font-medium text-sky-300 mb-1">Your Answer:</label>
                          <textarea
                            value={humanEvaluations[qType]?.[i] || ""}
                            onChange={(e) => handleHumanEvaluationChange(qType, i, e.target.value)}
                            placeholder="Enter your answer here..."
                            className="w-full p-2 text-sm rounded bg-slate-800 border border-slate-600 text-white focus:ring-1 focus:ring-sky-500 outline-none"
                            rows="3"
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className='flex justify-between mt-6'>
              <button 
                onClick={() => setStep(3)}
                className='px-6 py-2 bg-slate-700 text-gray-200 rounded-lg hover:bg-slate-600 transition'
              >
                ← Back
              </button>
              <button 
                onClick={submitHumanEvaluations}
                className='px-8 py-2 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-lg shadow hover:opacity-90 transition font-bold'
              >
                Continue to Metric Evaluation →
              </button>
            </div>
          </motion.div>
        )}

        {/* --- STEP 5: METRIC EVALUATION (Evaluation Mode) --- */}
        {appMode === 'evaluate' && step === 5 && (
          <motion.div
            key='evaluate-step5'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-4xl shadow-2xl border border-slate-700'
          >
            <h2 className='text-2xl font-bold text-sky-300 mb-6 flex items-center'>
              <BarChart2 className='w-6 h-6 mr-2 text-sky-400' />
              Metric-Based Evaluation
            </h2>

            <div className="mb-4 flex flex-col md:flex-row gap-4 items-start md:items-center">
              {/* Metric Select */}
              <div className="flex items-center gap-3 bg-slate-900/50 p-3 rounded-lg border border-slate-700 flex-grow">
                <label className="text-sm font-medium text-gray-300">Evaluation Metric:</label>
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  className="bg-slate-800 border border-slate-600 text-white text-sm rounded p-2 focus:ring-1 focus:ring-sky-500 outline-none w-full"
                >
                  {metricOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Upload Reference File */}
              <div className="flex items-center gap-3 bg-slate-900/50 p-3 rounded-lg border border-slate-700 flex-grow">
                 <label htmlFor="ref-upload" className="text-sm font-medium text-gray-300 cursor-pointer flex items-center gap-2">
                    <UploadCloud className="w-4 h-4 text-sky-400"/> 
                    {referenceFile ? "Change Reference" : "Upload Reference (Optional)"}
                 </label>
                 <input 
                    type="file" 
                    id="ref-upload"
                    accept=".pdf,.txt,.png,.jpg"
                    onChange={handleReferenceFileChange}
                    className="hidden"
                 />
                 {referenceFile && (
                    <span className="text-xs text-green-400 bg-green-900/30 px-2 py-1 rounded border border-green-700/50 flex items-center gap-1">
                       <CheckCircle className="w-3 h-3" /> {referenceFile.name.substring(0, 20)}...
                    </span>
                 )}
              </div>
            </div>

            {classNum && classNum !== "Custom" && !referenceFile && (
              <div className="mb-4 bg-green-900/20 border border-green-700/30 rounded-lg p-4">
                <p className="text-sm text-green-200">
                  <strong>📚 Using Textbook Reference:</strong> Questions will be compared against reference Q&A from the stored textbook (Class {classNum} - {subject}).
                </p>
              </div>
            )}
            
            {referenceFile && (
               <div className="mb-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                <p className="text-sm text-blue-200">
                  <strong>📄 Using Uploaded Reference:</strong> Questions will be compared against Q&A from: <b>{referenceFile.name}</b>
                </p>
              </div>
            )}

            {!referenceFile && (!classNum || classNum === "Custom") && (
              <div className="mb-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg p-4">
                <p className="text-sm text-yellow-200">
                  <strong>⚠️ No Reference Available:</strong> The system will attempt to generate answers from the context. Upload a reference file for better evaluation accuracy.
                </p>
              </div>
            )}

            <div className='bg-slate-900/70 text-white rounded-xl p-6 shadow-inner border border-slate-700 h-[500px] overflow-y-auto text-left'>
              {quizData && Object.keys(quizData).map(qType => (
                <div key={qType} className="mb-6">
                  <h3 className="text-xl font-bold text-sky-400 capitalize mb-3">{qType.replace(/_/g, ' ')}</h3>
                  <ul className="list-decimal list-inside space-y-6">
                    {quizData[qType]?.map((item, i) => (
                      <li key={i} className="space-y-3 pb-4 border-b border-slate-700/50 last:border-b-0">
                        <p className="whitespace-pre-line text-gray-200">{item.question}</p>
                        
                        {humanEvaluations[qType]?.[i] && (
                          <div className="mt-2 p-2 bg-blue-900/20 rounded border border-blue-700/30">
                            <span className="text-xs text-blue-300 font-bold">Your Answer:</span>
                            <p className="text-xs text-blue-200 mt-1">{humanEvaluations[qType][i]}</p>
                          </div>
                        )}

                        <div className="mt-3 flex items-center gap-3">
                          {!item.score && !item.evaluating && (
                            <button 
                              onClick={() => handleMetricEvaluate(qType, i)}
                              className="text-xs px-3 py-1.5 bg-sky-600 hover:bg-sky-500 rounded text-white transition"
                            >
                              Evaluate with {
                                selectedMetric === "meteor"
                                  ? "METEOR"
                                  : selectedMetric.toUpperCase()
                              }
                            </button>
                          )}
                          
                          {item.evaluating && (
                            <div className="flex items-center gap-2 text-xs text-sky-400">
                              <Loader className="animate-spin w-3 h-3" /> Evaluating...
                            </div>
                          )}

                          {item.score != null && (
                            <div className="flex items-center gap-3">
                              <div className="text-sm font-bold"><ScoreBadge score={item.score} /></div>
                              {!item.evaluating && (
                                <button 
                                  title="Re-evaluate with selected metric"
                                  onClick={() => handleMetricEvaluate(qType, i)}
                                  className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded-full transition"
                                >
                                  <RotateCcw className="w-3 h-3 text-sky-300" />
                                </button>
                              )}
                            </div>
                          )}
                        </div>

                        {item.score != null && item.bert_answer && (
                          <div className="mt-2 p-2 bg-slate-800/60 rounded border border-slate-700/50">
                            <p className="text-xs text-gray-400">{item.bert_answer}</p>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            {finalResults && finalResults.overallRemarks && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6"
              >
                <h3 className="text-xl font-bold text-sky-300 mb-3 flex items-center gap-2">
                  <ClipboardCheck className="w-5 h-5" />
                  Expert Remarks Summary
                </h3>
                <div className="bg-slate-900/70 rounded-xl p-6 border border-slate-700 max-h-48 overflow-y-auto">
                  <p className="whitespace-pre-line text-gray-300">{finalResults.overallRemarks}</p>
                </div>
              </motion.div>
            )}

            <div className='flex justify-between mt-6'>
              <button 
                onClick={() => setStep(4)}
                className='px-6 py-2 bg-slate-700 text-gray-200 rounded-lg hover:bg-slate-600 transition'
              >
                ← Back to Human Eval
              </button>
              
              <div className="flex gap-3">
                <button 
                  onClick={goToRatingPage}
                  className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg shadow hover:opacity-90 transition flex items-center gap-2'
                >
                  <Users className="w-4 h-4" /> Expert Difficulty Rating
                </button>

                <button 
                  onClick={resetToModeSelection}
                  className='px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg shadow hover:opacity-90 transition'
                >
                  🔄 New Session
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* --- STEP 6: EXPERT RATING (Evaluation Mode) --- */}
        {appMode === 'evaluate' && step === 6 && (
          <motion.div
            key='evaluate-step6'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className='relative bg-slate-800/80 backdrop-blur-xl rounded-2xl p-10 w-full max-w-6xl shadow-2xl border border-slate-700'
          >
            <h2 className='text-2xl font-bold mb-6 text-sky-300 flex items-center'>
              <Users className='w-6 h-6 mr-2 text-sky-400' />
              Expert Difficulty Rating
            </h2>
            
            {error && <p className="text-red-400 mb-4 text-left">{error}</p>}
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[75vh] min-h-0">
              <div className="flex flex-col h-full min-h-0">
                <h3 className="text-xl font-bold text-sky-400 mb-3 flex items-center gap-2">
                  <FileText className="w-5 h-5" /> 
                  Context Source
                </h3>
                <div className='bg-slate-900/70 text-white rounded-xl p-6 shadow-inner border border-slate-700 overflow-y-auto text-left flex-grow min-h-0'>
                  <p className="whitespace-pre-line text-gray-300 text-sm">{contextText || "No context loaded."}</p>
                </div>
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={currentInvigilatorId}
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -50 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-col h-full min-h-0"
                >
                  <InvigilatorRatingColumn
                    invigilatorId={`invigilator${currentInvigilatorId}`}
                    title={`Expert ${currentInvigilatorId}`}
                    quizData={quizData}
                    ratings={invigilatorRatings}
                    onChange={handleRatingChange}
                    onRemarkChange={handleRemarkChange}
                  />
                </motion.div>
              </AnimatePresence>
            </div>

            <div className='flex justify-between items-center mt-6 w-full'>
              <button
                onClick={handleInvigilatorBack}
                className='px-6 py-2 bg-slate-600 text-gray-200 rounded-lg hover:bg-slate-500 transition flex items-center gap-2'
              >
                <ArrowLeft className="w-4 h-4" />
                {currentInvigilatorId > 1 ? "Previous Expert" : "Back to Metrics"}
              </button>
              
              <div className="flex gap-3">
                  <button
                    onClick={handleSubmitRatings}
                    disabled={loading}
                    className='px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow hover:opacity-90 transition flex items-center gap-2 disabled:opacity-50'
                  >
                    {loading ? <Loader className="animate-spin w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
                    Finish & Submit
                  </button>

                  <button
                    onClick={handleAddInvigilator}
                    disabled={loading}
                    className='px-6 py-2 bg-gradient-to-r from-sky-500 to-blue-500 text-white rounded-lg shadow hover:opacity-90 transition flex items-center gap-2 disabled:opacity-50'
                  >
                    {invigilatorRatings[`invigilator${currentInvigilatorId+1}`] ? "Next Expert" : "Add Expert"}
                    <ArrowRight className="w-4 h-4" />
                  </button>
              </div>
            </div>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
}