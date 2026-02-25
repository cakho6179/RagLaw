/**
 * RagLaw API Client
 * ==================
 * 
 * Centralized API client with JWT auth support.
 * All endpoints use /api/v1/ prefix for new features.
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const API_V1 = `${API_BASE}/api/v1`;

// ============ Token Management ============

let accessToken = localStorage.getItem('raglaw_token') || null;

export function setToken(token) {
    accessToken = token;
    if (token) {
        localStorage.setItem('raglaw_token', token);
    } else {
        localStorage.removeItem('raglaw_token');
    }
}

export function getToken() {
    return accessToken;
}

export function clearAuth() {
    accessToken = null;
    localStorage.removeItem('raglaw_token');
    localStorage.removeItem('raglaw_user');
}

// ============ Request Helper ============

async function request(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
    };

    if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
    }

    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
        clearAuth();
        window.dispatchEvent(new Event('auth:expired'));
    }

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `Request failed: ${response.status}`);
    }

    return response.json();
}

// ============ Auth API ============

export const authApi = {
    async register(username, password, email = null) {
        const data = await request(`${API_V1}/auth/register`, {
            method: 'POST',
            body: JSON.stringify({ username, password, email }),
        });
        setToken(data.access_token);
        localStorage.setItem('raglaw_user', JSON.stringify(data.user));
        return data;
    },

    async login(username, password) {
        const data = await request(`${API_V1}/auth/login`, {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
        setToken(data.access_token);
        localStorage.setItem('raglaw_user', JSON.stringify(data.user));
        return data;
    },

    async getMe() {
        return request(`${API_V1}/auth/me`);
    },

    logout() {
        clearAuth();
    },
};

// ============ Sessions API ============

export const sessionsApi = {
    async list() {
        return request(`${API_V1}/sessions`);
    },

    async create(title = '') {
        return request(`${API_V1}/sessions`, {
            method: 'POST',
            body: JSON.stringify({ title }),
        });
    },

    async get(sessionId) {
        return request(`${API_V1}/sessions/${sessionId}`);
    },

    async update(sessionId, title) {
        return request(`${API_V1}/sessions/${sessionId}`, {
            method: 'PUT',
            body: JSON.stringify({ title }),
        });
    },

    async delete(sessionId) {
        return request(`${API_V1}/sessions/${sessionId}`, {
            method: 'DELETE',
        });
    },
};

// ============ Chat API ============

export const chatApi = {
    async send(query, sessionId = null, options = {}) {
        return request(`${API_V1}/chat`, {
            method: 'POST',
            body: JSON.stringify({
                query,
                session_id: sessionId,
                model: options.model || 'auto',
                mode: options.mode || 'flash',
                use_rag: options.use_rag !== false,
                top_k: options.top_k || 5,
            }),
        });
    },

    // Legacy endpoint (backward compat)
    async sendLegacy(query, sessionId = null, options = {}) {
        return request(`${API_BASE}/api/chat`, {
            method: 'POST',
            body: JSON.stringify({
                query,
                session_id: sessionId,
                model: options.model || 'auto',
                mode: options.mode || 'flash',
                use_rag: options.use_rag !== false,
                top_k: options.top_k || 5,
            }),
        });
    },
};

// ============ Documents API ============

export const documentsApi = {
    async list(page = 1, pageSize = 20, docType = null) {
        let url = `${API_V1}/documents?page=${page}&page_size=${pageSize}`;
        if (docType) url += `&doc_type=${encodeURIComponent(docType)}`;
        return request(url);
    },

    async search(query, page = 1, pageSize = 20, docType = null) {
        let url = `${API_V1}/documents/search?q=${encodeURIComponent(query)}&page=${page}&page_size=${pageSize}`;
        if (docType) url += `&doc_type=${encodeURIComponent(docType)}`;
        return request(url);
    },

    async get(docId) {
        return request(`${API_V1}/documents/${encodeURIComponent(docId)}`);
    },

    async getArticles(docId) {
        return request(`${API_V1}/documents/${encodeURIComponent(docId)}/articles`);
    },

    async getTypes() {
        return request(`${API_V1}/documents/types`);
    },

    async getStats() {
        return request(`${API_V1}/documents/stats`);
    },
};

// ============ Health API ============

export const healthApi = {
    async check() {
        return request(`${API_BASE}/api/health`);
    },
};

export default {
    auth: authApi,
    sessions: sessionsApi,
    chat: chatApi,
    documents: documentsApi,
    health: healthApi,
};
