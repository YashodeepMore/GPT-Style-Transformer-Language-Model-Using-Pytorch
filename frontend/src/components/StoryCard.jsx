export default function StoryCard({ story, index }) {
  return (
    <article className="story-card">
      <div className="story-number">Story {index + 1}</div>
      <p>{story}</p>
    </article>
  );
}
