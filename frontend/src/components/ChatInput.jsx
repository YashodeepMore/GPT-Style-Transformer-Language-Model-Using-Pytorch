import { useState } from "react";

export default function ChatInput({ onSubmit, loading }) {
  const [prompt, setPrompt] = useState("Once upon a time");

  const submitPrompt = () => {
    if (!loading && prompt.trim()) {
      onSubmit(prompt);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitPrompt();
    }
  };

  return (
    <div className="composer">
      <textarea
        aria-label="Story prompt"
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Begin your story..."
        rows={3}
        disabled={loading}
      />
      <div className="composer-footer">
        <span>Enter to generate · Shift + Enter for a new line</span>
        <button
          type="button"
          onClick={submitPrompt}
          disabled={loading || !prompt.trim()}
        >
          {loading ? "Writing..." : "Generate"}
        </button>
      </div>
    </div>
  );
}
