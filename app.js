document.addEventListener("DOMContentLoaded", () => {
    const API_ENDPOINT = "http://127.0.0.1:5000/api/recommend";
    const API_USERS    = "http://127.0.0.1:5000/api/users";
    const API_PROFILE  = (uid) => `http://127.0.0.1:5000/api/user_profile/${uid}`;

    const topkSlider   = document.getElementById("topk");
    const topkDisplay  = document.getElementById("topk-v");
    const btnSubmit    = document.querySelector(".btn-go");
    const btnClearHist = document.getElementById("btn-clear-history");
    const welcomeArea  = document.getElementById("welcome");
    const resultsArea  = document.getElementById("results");
    const historyPanel = document.getElementById("history-panel");
    const userSelect   = document.getElementById("user-select");
    const userInfoText = document.getElementById("user-info");

    let selectedUserId    = null;
    let adaptiveThreshold = 0.30;
    let history  = JSON.parse(localStorage.getItem("foodrec_history") || "[]");
    // seenIds: oturum boyunca gösterilen tarif ID'leri — sayfa kapanınca sıfırlanır
    let seenIds  = JSON.parse(sessionStorage.getItem("foodrec_seen") || "[]");

    // ── Önerileri Sunucudan Çeken Ana Fonksiyon ──
    async function fetchRecommendations(isManualClick) {
        if (btnSubmit) {
            btnSubmit.textContent = "İşleniyor... ⏳";
            btnSubmit.disabled = true;
        }

        const likes = getSelected("#like-tags .tag.on");
        const historyWithRec = [...history];

        const payload = {
            meal:        document.querySelector("#meal-tags .tag.on")?.textContent.trim()   || "Lunch",
            season:      document.querySelector("#season-tags .tag.on")?.textContent.trim() || "Summer",
            max_minutes: parseInt(document.querySelector("#time-tags .tag.on")?.dataset.value || "999", 10),
            likes,
            dislikes:    getSelected("#dis-tags .tag.on"),
            history:     historyWithRec,
            topk:        parseInt(topkSlider.value, 10),
            user_id:     selectedUserId,
            seen_ids:    seenIds,   // backend bu ID'leri hariç tutar → her seferinde farklı liste
        };

        try {
            const res  = await fetch(API_ENDPOINT, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(payload),
            });
            const data = await res.json();

            // Yeni gelen ID'leri seen listesine ekle
            // 80 eşiğine ulaşınca en eski yarısını temizle — havuz daralmasını önler
            if (data.recommendations?.length) {
                seenIds = [...seenIds, ...data.recommendations.map(r => r.id)];
                if (seenIds.length > 80) {
                    seenIds = seenIds.slice(-40); // en yeni 40'ı tut
                }
                sessionStorage.setItem("foodrec_seen", JSON.stringify(seenIds));
            }

            if (isManualClick) {
                const allLikes = Array.from(document.querySelectorAll("#like-tags .tag.on"))
                    .map(t => t.textContent.trim());
                if (allLikes.length > 0) addToHistory(allLikes);
            } else {
                // Profil otomatik yüklemesinde: sadece profil tag'lerini (tag-dynamic) history'e ekle
                // Bu sayede profilden gelen malzemeler doygunluk hesabına dahil olur
                const profileTags = Array.from(document.querySelectorAll("#like-tags .tag.on.tag-dynamic"))
                    .map(t => t.textContent.trim());
                if (profileTags.length > 0) addToHistory(profileTags);
            }
            renderHistoryPanel(data.saturation || []);
            renderResults(data.recommendations || []);
        } catch (err) {
            console.error(err);
            alert("Sunucu bağlantı hatası!");
        } finally {
            if (btnSubmit) {
                btnSubmit.textContent = "Öneri Getir 🍳";
                btnSubmit.disabled = false;
            }
        }
    }

    // ── Kullanıcı dropdown'u doldur ──
    async function loadUsers() {
        try {
            const res  = await fetch(API_USERS);
            const data = await res.json();
            if (!data.success || !data.users.length) return;
            data.users.forEach(uid => {
                const opt = document.createElement("option");
                opt.value       = uid;
                opt.textContent = `👤 Kullanıcı #${uid}`;
                userSelect.appendChild(opt);
            });
        } catch {
            const demoUsers = [47892, 12045, 8834, 23156, 5541];
            demoUsers.forEach(uid => {
                const opt = document.createElement("option");
                opt.value       = uid;
                opt.textContent = `👤 Kullanıcı #${uid} (demo)`;
                userSelect.appendChild(opt);
            });
        }
    }
    loadUsers(); 

    // ── Kullanıcı ID'si seçildiğinde tetiklenen kısım ──
    if (userSelect) {
        userSelect.addEventListener("change", async () => {
            const val = userSelect.value;
            if (!val) {
                selectedUserId    = null;
                adaptiveThreshold = 0.30;
                seenIds           = [];
                sessionStorage.removeItem("foodrec_seen");
                clearAllLikeTags();
                if (userInfoText) userInfoText.textContent = "";
                return;
            }

            selectedUserId = parseInt(val, 10);
            if (userInfoText) userInfoText.textContent = "Profil yükleniyor...";

            try {
                const res  = await fetch(API_PROFILE(selectedUserId));
                const data = await res.json();
                if (!data.success) throw new Error(data.error);

                adaptiveThreshold = data.adaptive_threshold;
                clearAllLikeTags();
                applyProfileTags(data.top_ingredients);

                if (userInfoText) {
                    userInfoText.textContent =
                        `${data.liked_count} beğenilen tarif · ${data.n_interactions} etkileşim · ` +
                        `Kişisel eşik: %${Math.round(data.adaptive_threshold * 100)}`;
                }

                // 🌟 [OTOMATİK SUNUM]: ID algılandığı an saf CF önerilerini gürültüsüz getirir.
                setTimeout(() => {
                    fetchRecommendations(false); 
                }, 150);

            } catch (err) {
                if (userInfoText) userInfoText.textContent = "Profil yüklenemedi.";
                console.error(err);
            }
        });
    }

    // ── Butona Basıldığında Tetiklenen Kısım (Değiştirme/Yenileme) ──
    if (btnSubmit) {
        btnSubmit.addEventListener("click", () => {
            fetchRecommendations(true); // true = gürültüyü/karıştırmayı aç, listeyi değiştir.
        });
    }

    // ── Yardımcı Fonksiyonlar ──
    function clearAllLikeTags() {
        document.querySelectorAll("#like-tags .tag").forEach(t => {
            t.classList.remove("on", "tag-dynamic");
            t.title = "";
        });
        // Dinamik olarak eklenen profil tag'lerini kaldır
        document.querySelectorAll("#like-tags .tag-dynamic").forEach(t => t.remove());
        // Profil hint'ini kaldır
        const hint = document.getElementById("profile-hint");
        if (hint) hint.remove();
    }

    function applyProfileTags(ingredients) {
        const grid = document.getElementById("like-tags");
        if (!grid) return;

        // Profil bilgi notu — bir kez ekle
        if (!document.getElementById("profile-hint")) {
            const hint = document.createElement("p");
            hint.id = "profile-hint";
            hint.className = "profile-hint-text";
            hint.textContent = "★ Geçmiş etkileşimlerinden · İstersen değiştirebilirsin";
            grid.parentElement.insertBefore(hint, grid);
        }

        ingredients.forEach(ing => {
            let found = false;
            grid.querySelectorAll(".tag").forEach(tag => {
                if (tag.textContent.trim().toLowerCase() === ing.toLowerCase()) {
                    tag.classList.add("on", "tag-dynamic");
                    tag.title = "Profilinden geldi · tıklayarak kaldırabilirsin";
                    found = true;
                }
            });
            if (!found) {
                const newTag = document.createElement("div");
                newTag.className   = "tag t-like on tag-dynamic";
                newTag.textContent = ing;
                newTag.title       = "Profilinden geldi · tıklayarak kaldırabilirsin";
                // Ayrı listener EKLEME — initTagGroup'un event delegation'ı zaten yakalar
                grid.appendChild(newTag);
            }
        });
    }

    function addToHistory(likes) {
        const now = Date.now();
        likes.forEach(name => history.push({ name: name.toLowerCase().trim(), ts: now }));
        const cutoff = now - 30 * 86_400_000;
        history = history.filter(h => h.ts >= cutoff);
        saveHistory();
        renderHistoryPanel(null);
    }

    function saveHistory() { localStorage.setItem("foodrec_history", JSON.stringify(history)); }

    function renderHistoryPanel(saturation) {
        if (!historyPanel) return;
        if (history.length === 0 && !saturation) {
            historyPanel.innerHTML = '<p class="no-history">Henüz geçmiş yok. İlk öneriyi aldıktan sonra burada görünecek.</p>';
            return;
        }
        let rows = "";
        if (saturation && saturation.length > 0) {
            rows = saturation.map(s => {
                const clampedPct = Math.min(s.penalty_pct, Math.round(adaptiveThreshold * 100));
                const badge = s.heavy ? `<span class="sat-badge sat-high">Çok sık!</span>` : `<span class="sat-badge sat-mid">Tekrarlı</span>`;
                return `<div class="sat-row"><span class="sat-ing">${s.ingredient}</span>${badge}<span class="sat-info">${s.approx_times}x seçildi · -%${clampedPct} ceza</span></div>`;
            }).join("");
        } else {
            const counts = {};
            history.forEach(h => { counts[h.name] = (counts[h.name] || 0) + 1; });
            rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name, cnt]) => `
                <div class="sat-row"><span class="sat-ing">${name}</span><span class="sat-info">${cnt}x seçildi</span></div>`).join("");
        }
        historyPanel.innerHTML = `<div class="sat-title">Geçmişin & Doygunluk Durumu</div><div class="sat-list">${rows}</div>`;
    }
    renderHistoryPanel(null);

    if (btnClearHist) {
        btnClearHist.addEventListener("click", () => {
            history = [];
            seenIds = [];
            saveHistory();
            sessionStorage.removeItem("foodrec_seen");
            renderHistoryPanel(null);
        });
    }

    if (topkSlider && topkDisplay) {
        topkSlider.addEventListener("input", e => { topkDisplay.textContent = e.target.value; });
    }

    const initTagGroup = (gridId, isMultiSelect = true) => {
        const grid = document.getElementById(gridId);
        if (!grid) return;
        grid.addEventListener("click", e => {
            const tag = e.target.closest(".tag");
            if (!tag) return;
            if (!isMultiSelect) grid.querySelectorAll(".tag").forEach(t => t.classList.remove("on"));
            tag.classList.toggle("on");
        });
    };
    initTagGroup("meal-tags",   false);
    initTagGroup("season-tags", false);
    initTagGroup("like-tags",   true);
    initTagGroup("dis-tags",    true);
    initTagGroup("time-tags",   false);

    const getSelected = sel => Array.from(document.querySelectorAll(sel)).map(t => t.textContent.trim());

    window.showRecipeSteps = (b64) => {
        try {
            const steps = JSON.parse(decodeURIComponent(atob(b64)));

            // Varsa eski modalı kaldır
            document.getElementById("recipe-modal")?.remove();

            const overlay = document.createElement("div");
            overlay.id = "recipe-modal";
            overlay.style.cssText = `
                position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:9999;
                display:flex;align-items:center;justify-content:center;padding:20px;
            `;

            const box = document.createElement("div");
            box.style.cssText = `
                background:#fff;border-radius:20px;padding:32px;max-width:560px;width:100%;
                max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.25);
                position:relative;font-family:'Poppins',sans-serif;
            `;

            const closeBtn = document.createElement("button");
            closeBtn.textContent = "✕";
            closeBtn.style.cssText = `
                position:absolute;top:16px;right:16px;background:none;border:none;
                font-size:1.2rem;cursor:pointer;color:#6b7280;
            `;
            closeBtn.onclick = () => overlay.remove();

            const title = document.createElement("h3");
            title.textContent = "📋 Tarif Adımları";
            title.style.cssText = "margin-bottom:18px;color:#16a34a;font-size:1.15rem;";

            const list = document.createElement("ol");
            list.style.cssText = "padding-left:20px;display:flex;flex-direction:column;gap:10px;";
            steps.forEach(s => {
                const li = document.createElement("li");
                li.textContent = s;
                li.style.cssText = "font-size:.88rem;color:#374151;line-height:1.55;";
                list.appendChild(li);
            });

            box.appendChild(closeBtn);
            box.appendChild(title);
            box.appendChild(list);
            overlay.appendChild(box);
            overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
            document.body.appendChild(overlay);
        } catch {
            alert("Adımlar yüklenemedi.");
        }
    };

    // ── Sonuçları Ekrana Basan Güncellenmiş Fonksiyon ──
    function renderResults(recipes) {
        if (!welcomeArea || !resultsArea) return;
        welcomeArea.classList.add("hidden");
        resultsArea.classList.remove("hidden");

        const thresholdBadge = selectedUserId
            ? `<div class="threshold-info">👤 Kullanıcı #${selectedUserId} · Kişisel Doygunluk Eşiği: <strong>%${Math.round(adaptiveThreshold * 100)}</strong></div>`
            : `<div class="threshold-info">👤 Misafir · Varsayılan Eşik: <strong>%30</strong></div>`;

        resultsArea.innerHTML = thresholdBadge + recipes.map(r => {
            const b64      = btoa(encodeURIComponent(JSON.stringify(r.steps)));
            const penBadge = r.penalty > 0 ? `<span class="pen-badge">-%${Math.round(Math.min(r.penalty, adaptiveThreshold) * 100)} doygunluk cezası · ${r.penalized_by.join(", ")}</span>` : "";
            return `
            <div class="recipe-card ${r.penalty > 0 ? 'card-penalized' : ''}">
                <div class="card-header">
                    <span class="meal-badge">⏰ ${r.meal_type}</span>
                    <span class="score-text">Skor: ${r.final_score}</span>
                </div>
                <h3>${r.name}</h3>
                ${penBadge}
                <p style="margin:6px 0;font-size:.85rem;color:var(--text-light)">
                    <strong>Süre:</strong> ${r.minutes} dk &nbsp;|&nbsp;
                    <strong>Malzemeler:</strong> ${r.ingredients.slice(0,5).join(", ")}…
                </p>
                <button class="btn-tarif" onclick="window.showRecipeSteps('${b64}')">Tarifi Gör</button>
                
                <div class="penalty-grid" style="grid-template-columns: repeat(3, 1fr); gap: 5px; font-size: 0.78rem; text-align: center; line-height: 1.3;">
                    <div><strong>Malzeme Uyumu (CB)</strong><br>${r.cb_score}</div>
                    <div><strong>Ortak Zevk (CF)</strong><br>${r.cf_score}</div>
                    <div><strong>Sıkılma Cezası (Ceza)</strong><br>-${r.penalty}</div>
                </div>
            </div>`;
        }).join("");
    }
});