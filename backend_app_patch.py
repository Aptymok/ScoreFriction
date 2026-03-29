# ============================================================
# PATCH PARA backend/app.py
# Agregar al final del archivo, antes del if __name__ == '__main__':
# ============================================================

@app.route('/state', methods=['GET'])
def get_state():
    """Estado MIHM compacto para el hub frontend."""
    mihm.process_delayed_updates()
    return jsonify({
        'ihg':        mihm.state.get('ihg', -0.62),
        'nti':        mihm.state.get('nti', 0.351),
        'r':          mihm.state.get('r', 0.45),
        'cff':        mihm.state.get('cff', 0.0),
        'irc':        mihm.irc,
        'cost_j':     mihm.cost_function(),
        'timestamp':  datetime.utcnow().isoformat(),
    })


@app.route('/pm/event', methods=['POST'])
def pm_event():
    """Bridge entre PM frontend y motor MIHM.
    El PM llama esto cuando ocurren eventos operativos.
    El usuario nunca ve IHG/NTI — solo ve el efecto en Momentum.
    """
    ev = request.json.get('type', '')
    if ev == 'project_done':
        u, J = mihm.apply_delta({'nti': 0.06, 'r': 0.04, 'ihg': 0.08},
                                action='pm:project_done')
    elif ev == 'project_created':
        count = int(request.json.get('count', 1))
        pressure = min(0.8, count * 0.06)
        u, J = mihm.apply_delta({'ihg': -pressure * 0.06},
                                action='pm:project_created')
    elif ev == 'task_done':
        u, J = mihm.apply_delta({'r': 0.02, 'nti': 0.01},
                                action='pm:task_done')
    elif ev == 'task_late':
        u, J = mihm.apply_delta({'r': -0.03},
                                action='pm:task_late')
    elif ev == 'audit':
        answers = request.json.get('answers', {})
        delta = {}
        if answers.get('ts') == 'solo':   delta['ihg'] = -0.25
        if answers.get('ts') == '11+':    delta['ihg'] = 0.1
        if answers.get('up') == 'yes-c':
            delta.update({'nti': -0.12, 'r': -0.08})
        if answers.get('up') == 'no':
            delta['nti'] = 0.06
        u, J = mihm.apply_delta(delta, action='pm:audit')
    else:
        u, J = 0.0, mihm.cost_function()

    mihm.meta_control()
    db.save_state(mihm.state, mihm.irc, f'pm_event:{ev}', J or mihm.cost_function())

    return jsonify({
        'state':  mihm.state,
        'cost_j': J or mihm.cost_function(),
        'u':      u,
        'irc':    mihm.irc,
    })


@app.route('/scraping', methods=['POST'])
def scraping():
    """Tendencias sociales para el PM (campañas, analytics)."""
    data   = request.get_json() or {}
    genre  = data.get('genre', 'pop')
    query  = data.get('query', genre)
    try:
        results = scraper.scrape_genius(query)
        if not results:
            results = [
                {'title': f'{genre.title()} trending', 'keywords': [genre, 'viral', 'loop'], 'artist': 'scraper'},
            ]
    except Exception:
        results = [{'title': f'{genre.title()} emergente', 'keywords': [genre], 'artist': 'fallback'}]

    return jsonify({'status': 'ok', 'genre': genre, 'results': results})


@app.route('/chat', methods=['POST'])
def chat_proxy():
    """Proxy para chat IA cuando el frontend no tiene Groq key directo."""
    data     = request.get_json() or {}
    messages = data.get('messages', [])
    if not messages or not groq.api_key:
        return jsonify({'response': 'Backend activo. Configura GROQ_API_KEY en .env para respuestas completas.'})
    try:
        response = groq._call(messages, temperature=0.7, max_tokens=1400)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error LLM: {e}'}), 500
