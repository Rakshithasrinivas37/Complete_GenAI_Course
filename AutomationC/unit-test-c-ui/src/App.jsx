import { useState } from "react";
import "./App.css";

function App() {
  const [cFileName, setCFileName] = useState("");
  const [cCode, setCCode] = useState("");
  const [testCode, setTestCode] = useState("");
  const [promptMode, setPromptMode] = useState(false);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setCFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      setCCode(e.target.result);
    };
    reader.readAsText(file);
  };

  const handleGenerate = () => {
    // 🔥 Placeholder for API call
    setTestCode(`#include "unity.h"

void test_addition() {
    TEST_ASSERT_EQUAL(5, add(2, 3));
}

void test_multiplication() {
    TEST_ASSERT_EQUAL(12, multiply(3, 4));
}`);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Automation Unit Test Case Generator</h1>
        <p>Upload C code → Get Unit Tests Automatically</p>
      </header>

      <div className="container">
        {/* LEFT PANEL */}
        <div className="panel">
          <h2>C Source Code</h2>

          <div className="controls">
            <label className="file-upload">
              Choose C File
              <input type="file" accept=".c" onChange={handleFileUpload} />
            </label>

            <div className="toggle">
              <span>Prompt Mode</span>
              <input
                type="checkbox"
                checked={promptMode}
                onChange={() => setPromptMode(!promptMode)}
              />
            </div>

            <button
              className="generate-btn"
              disabled={!cCode}
              onClick={handleGenerate}
            >
              Generate Unit Tests
            </button>
          </div>

          <textarea
            className="code-editor"
            value={cCode}
            placeholder="C source code will appear here..."
            readOnly
          />
        </div>

        {/* RIGHT PANEL */}
        <div className="panel">
          <h2>Generated Unit Test Cases</h2>

          <textarea
            className="code-editor"
            value={testCode}
            placeholder="Generated unit tests will appear here..."
            readOnly
          />
        </div>
      </div>
    </div>
  );
}

export default App;
