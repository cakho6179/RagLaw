/**
 * RagLaw TypeScript Type Definitions
 * ====================================
 * Shared types for API communication and UI state.
 */

// ============ Auth Types ============

export interface User {
    id: number;
    username: string;
    email: string | null;
    created_at: string;
}

export interface AuthResponse {
    user: User;
    access_token: string;
    token_type: string;
}

export interface LoginRequest {
    username: string;
    password: string;
}

export interface RegisterRequest {
    username: string;
    password: string;
    email?: string;
}

// ============ Session Types ============

export interface ChatSession {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    message_count?: number;
}

export interface Message {
    id?: number;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    sources?: MessageSource[];
}

export interface MessageSource {
    document_id: string | null;
    document_title: string | null;
    article: string | null;
    clause: string | null;
    source_text: string;
    score: number;
}

export interface SessionDetail {
    session: ChatSession;
    messages: Message[];
}

// ============ Chat Types ============

export interface ChatRequest {
    query: string;
    session_id?: string;
    model?: string;
    mode?: 'flash' | 'pro';
    use_rag?: boolean;
    top_k?: number;
}

export interface ChatResponse {
    answer: string;
    session_id: string;
    citations: string[];
    sources: Record<string, any>[];
    confidence: number;
    model_used: string;
    mode: string;
}

// ============ Document Types ============

export interface LawDocument {
    id: string;
    title: string;
    document_type: string;
    official_number: string;
    issued_date: string;
    effective_date: string;
    snippet: string;
    article_count: number;
    source: string;
}

export interface LawArticle {
    article_id: string;
    title: string;
    text: string;
}

export interface DocumentDetail {
    id: string;
    title: string;
    document_type: string;
    official_number: string;
    issued_date?: string;
    effective_date?: string;
    signer?: string;
    articles?: LawArticle[];
    source: string;
}

export interface PaginatedDocuments {
    documents: LawDocument[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
    query?: string;
}
