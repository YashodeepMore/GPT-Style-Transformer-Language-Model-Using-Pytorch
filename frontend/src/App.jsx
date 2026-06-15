import { useState } from "react";
import axios from "axios";
import ChatInput from "./components/ChatInput";
import Loader from "./components/Loader";
import StoryCard from "./components/StoryCard";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function getErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (detail?.message) {
    return detail.message;
  }
  if (Array.isArray(detail)) {
    return detail[0]?.msg ?? "Please check your prompt and try again.";
  }
  if (error.code === "ERR_NETWORK") {
    return "Cannot reach the story server. Make sure the backend is running.";
  }
  return "Something went wrong while generating stories.";
}

export default function App() {
  const [stories, setStories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generateStories = async (prompt) => {
    setLoading(true);
    setError("");
    setStories([]);

    try {
      const response = await axios.post(`${API_URL}/generate`, { prompt });
      setStories(response.data.stories);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="eyebrow">CUSTOM TRANSFORMER</div>
        <h1>Turn a small beginning into a new story.</h1>
        <p>
          Give TinyStories an opening line and let the model imagine what
          happens next.
        </p>
      </section>

      <ChatInput onSubmit={generateStories} loading={loading} />

      <section className="results" aria-live="polite">
        {loading && <Loader />}
        {error && <div className="error-message">{error}</div>}
        {stories.map((story, index) => (
          <StoryCard key={`${index}-${story.slice(0, 24)}`} story={story} index={index} />
        ))}
      </section>
    </main>
  );
}
