import { useState, useEffect } from 'react'
import './Evaluation.css'

function Evaluation() {
    const [evaluationType, setEvaluationType] = useState('mcq') // mcq | nli | syllogism
    const [results, setResults] = useState(null)
    const [isRunning, setIsRunning] = useState(false)
    const [progress, setProgress] = useState({ current: 0, total: 0 })
    const [selectedModel, setSelectedModel] = useState('gemini')

    const evaluationTypes = [
        { id: 'mcq', name: 'Trắc nghiệm', icon: '📝', description: 'Câu hỏi trắc nghiệm pháp luật' },
        { id: 'nli', name: 'NLI', icon: '🔍', description: 'Suy luận ngôn ngữ tự nhiên' },
        { id: 'syllogism', name: 'Syllogism', icon: '🧠', description: 'Suy luận logic pháp lý' }
    ]

    const runEvaluation = async () => {
        setIsRunning(true)
        setResults(null)
        setProgress({ current: 0, total: 0 })

        try {
            const res = await fetch('/api/evaluation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: evaluationType,
                    model: selectedModel,
                    limit: 50
                })
            })

            if (res.ok) {
                const data = await res.json()
                setResults(data)
            } else {
                throw new Error('Evaluation failed')
            }
        } catch (err) {
            setResults({ error: err.message })
        } finally {
            setIsRunning(false)
        }
    }

    return (
        <div className="evaluation-container">
            {/* Header */}
            <div className="eval-header glass-panel">
                <div className="header-info">
                    <h1>📊 KAISER Evaluation</h1>
                    <p>Đánh giá hiệu suất hệ thống RAG pháp luật</p>
                </div>
                <div className="header-controls">
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="model-select"
                    >
                        <option value="gemini">🌟 Gemini 2.5</option>
                        <option value="hf">🤖 Qwen 3B</option>
                    </select>
                </div>
            </div>

            {/* Evaluation Types */}
            <div className="eval-types">
                {evaluationTypes.map(type => (
                    <div
                        key={type.id}
                        className={`eval-type-card glass-panel ${evaluationType === type.id ? 'active' : ''}`}
                        onClick={() => setEvaluationType(type.id)}
                    >
                        <div className="type-icon">{type.icon}</div>
                        <div className="type-info">
                            <h3>{type.name}</h3>
                            <p>{type.description}</p>
                        </div>
                        {evaluationType === type.id && <div className="type-check">✓</div>}
                    </div>
                ))}
            </div>

            {/* Run Button */}
            <div className="eval-actions">
                <button
                    className="run-btn"
                    onClick={runEvaluation}
                    disabled={isRunning}
                >
                    {isRunning ? (
                        <>⏳ Đang đánh giá... ({progress.current}/{progress.total})</>
                    ) : (
                        <>🚀 Bắt đầu đánh giá</>
                    )}
                </button>
            </div>

            {/* Results */}
            {results && (
                <div className="eval-results glass-panel animate-fadeIn">
                    {results.error ? (
                        <div className="result-error">
                            <span className="error-icon">❌</span>
                            <p>{results.error}</p>
                        </div>
                    ) : (
                        <>
                            <div className="results-header">
                                <h2>📈 Kết quả đánh giá</h2>
                            </div>

                            <div className="stats-grid">
                                <div className="stat-card">
                                    <div className="stat-value">{results.accuracy || 0}%</div>
                                    <div className="stat-label">Độ chính xác</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{results.total || 0}</div>
                                    <div className="stat-label">Tổng câu hỏi</div>
                                </div>
                                <div className="stat-card correct">
                                    <div className="stat-value">{results.correct || 0}</div>
                                    <div className="stat-label">Đúng</div>
                                </div>
                                <div className="stat-card wrong">
                                    <div className="stat-value">{results.wrong || 0}</div>
                                    <div className="stat-label">Sai</div>
                                </div>
                            </div>

                            {results.details && results.details.length > 0 && (
                                <div className="results-details">
                                    <h3>Chi tiết kết quả</h3>
                                    <div className="details-table">
                                        {results.details.slice(0, 10).map((d, i) => (
                                            <div key={i} className={`detail-row ${d.correct ? 'correct' : 'wrong'}`}>
                                                <span className="detail-q">{d.question?.substring(0, 80)}...</span>
                                                <span className="detail-answer">
                                                    {d.correct ? '✓' : '✗'} {d.predicted} (đáp án: {d.actual})
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            )}

            {/* Empty State */}
            {!results && !isRunning && (
                <div className="eval-empty glass-panel">
                    <div className="empty-icon">📊</div>
                    <h3>Chọn loại đánh giá và bắt đầu</h3>
                    <p>Kết quả sẽ hiển thị tại đây</p>
                </div>
            )}
        </div>
    )
}

export default Evaluation
