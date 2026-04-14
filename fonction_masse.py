import math
import numpy as np 

battery_params = {
    "LFP": {
        "Li_%": 0.140, "Co_%": 0, "Ni_%": 0, "Cu_%": 0.340,"Graphite":1.120,
    },
    "NMC622": {
        "Li_%": 0.115, "Co_%": 0.190, "Ni_%": 0.620, "Cu_%": 0.230,"Graphite":1,
    },
    "NMC811": {
        "Li_%": 0.094, "Co_%": 0.050, "Ni_%": 0.693, "Cu_%": 0.220,"Graphite":1,
    },
    "LMFP": {
        "Li_%": 0.150, "Co_%": 0, "Ni_%": 0, "Cu_%": 0.280,"Graphite":1,
    },
    
}
magnet_params = {
    "NdFeB": 1.25,   # Tesla
    "SmCo": 1.05,
    "Ferrite": 0.40,
    "AlNiCo": 0.90
}
def to_python_scalar(val):
    """Convertit numpy arrays/scalaires en valeurs Python natives"""
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        else:
            return val.tolist()
    return val
def init():
    return True

def masse_composants_batt(batt,E_batt):
    dico_masse={
        "Li":E_batt*battery_params[batt]["Li_%"],
        "Co":E_batt*battery_params[batt]["Co_%"],
        "Ni":E_batt*battery_params[batt]["Ni_%"],
        "Cu":E_batt*battery_params[batt]["Cu_%"],
        "Graphite":E_batt*battery_params[batt]["Graphite"],
    }
    return dico_masse

def volume_aimants(subtype_val,R,L,g,bmax,pole_pairs,magnet):
    Bg = bmax #induction entrefer cible
    Br = magnet_params[magnet]
    mu_r=1000
    l_pont = 1e-3
    alpha=0.9
    if subtype_val=="msap_surface":
        lmag = 2 * (bmax**2 / (alpha*Br)**2) * (g+l_pont/mu_r)
        Vmag_unite = (lmag * 2 * math.pi * R * L) /(2* pole_pairs)
        Vmag_tot = Vmag_unite *2* pole_pairs
    if subtype_val == "msap_enterrer_radial":
        l_pont = 1e-3
        alpha=0.9 #leakage flux
        k_reel = 1.3 #facteur de reluctance
        l_mag =  2*(Bg / (alpha * Br)) * (g + l_pont/mu_r)
        Vmag_unite = (l_mag * 2 * math.pi * R * L) / (2*pole_pairs)
        Vmag_tot = (Vmag_unite * 2*pole_pairs)/k_reel
    if subtype_val == "msap_enterrer_V":
        l_pont = 1e-3
        alpha=0.9 #leakage flux
        k_reel = 1.5 #facteur de reluctance
        l_mag =  2*(Bg / (alpha * Br)) * (g + l_pont/mu_r)
        Vmag_unite = (l_mag * 2 * math.pi * R * L) / (2*pole_pairs)
        Vmag_tot = (Vmag_unite * 2*pole_pairs)/k_reel
    if subtype_val == "msap_enterrer_axial":
        #ici on est sur une structure à simple entrefer 
        mu0 = 4*math.pi*1e-7
        R_ext = R    # m, rayon ext disque = rayon entrefer
        R_int = 50e-3     # m, rayon int (hub)
        R_entrefer = 2*g / (mu0 * math.pi*(R_ext**2 - R_int**2)/(2*pole_pairs))
        k = 1.9
        alpha = 0.85
        mufer_sat = 500*mu0
        Bg = bmax / (1 +  0.04)
        l_mag = k * (Bg / (alpha * Br)) * (2*g)
        S_pole = math.pi*(R_ext**2 - R_int**2) / 2*pole_pairs
        Vmag_unite = l_mag * S_pole
        Vmag_tot = Vmag_unite * 4 * pole_pairs
    return Vmag_tot
def cuivre_stator(P_traction,P_alim,V,bmax,R,L,p,Couple):
    #dans le cas d'une machine synchrone 
    I_ph=(P_alim*1e3)/(math.sqrt(3)*V) #%Courant dans une phase en négligeant le cos(phi)
    J=16e6 #densité de courant admissible entre 3 et 6 A/mm^2 selon le refroidissement 
    S_cu = I_ph/J 
    #Normalement on dimensionne en fonction du courant admissible et du courant de phase
    #Or pour les deux grandeurs on a pas bcp de données on prend donc un bobinage moyen 
    #S_cu = I_ph/J 
    #S_cu = 10e-6
    Flux_phase=bmax*math.pi*2*R*L/(p)
    Omega_base= 52 # Nbase = 500 tr/min dans le cahier des charges
    f=p*Omega_base 
    K_w = 0.4 #facteur d'enroulement entre 0.3 et 0.4
    K_b = 0.7 #coeff de bobinage prenant en compte le facteur de puissance K_b =  E/V
    N_spire_phase=(K_b*V)/(math.sqrt(2)*f*Flux_phase*K_w)
    N_spires=3*N_spire_phase
    k= 0.8  #fonction du schéma d'enroulement entre 0.5 et 0.8
    # k facteur de géométrie de la tête d'enroulement
    
    l_tete=k*(30e-3)

    l_tot_spire = (2*L+2*l_tete)*N_spires
    V_cu_stator = l_tot_spire*S_cu
    P_cu_s=3*((l_tot_spire*0.017)/S_cu)*1e3
    return [V_cu_stator,P_cu_s]

def cuivre_rotor_ms(I_exc, bmax, R, L, p):
    # Densité courant
    J = 6  # A/mm² admissible
    # Section cuivre
    S_cu = I_exc * (1e-6) / J  # m²
    
    # Flux par pôle
    Flux_pole = bmax * (2 * math.pi * R * L) / p
    
    # Ampère-tour 
    kc = 0.7 #coefficient de carter fonction de l'ouverture d'encoche largeur dent  et entrefer 
    mu0 = 4 * math.pi * 1e-7
    AT_pole = bmax * L /(kc* mu0)  
    
    # Spires par pôle
    N_spires_pole = AT_pole / I_exc
    N_spires_total = p * N_spires_pole
    
    # Longueur tête
    k = 0.6
    #on prend une longueur de tete fixée à 30mm en moyenne
    l_tete = 30*1e-3
    
    # Longueur totale spires
    l_tot_spire = (2 * L + 2 * l_tete) * N_spires_total
    
    # Volume cuivre
    V_cu_rotor = l_tot_spire * S_cu
    return V_cu_rotor
def conducteur_rotor_mas(g,Ps_cu,R,L):
    #On calcule le volume de conducteur en fonction des pertes au rotor Pr = g*Ps_cu 
    # avec Ps_cu pertes cuivres au stator
    #g glissement
    #On peut choisir du cuivre ou de l'aluminium ici on prend du cuivre
    #On doit avoir Vrotor pour calculer les courants rotoriques
    #ρ_Cu = 0,017 Ω·mm²/m vs ρ_Al = 0,028 Ω·mm²/m
    p_Cu = 0.017*1e-3
    J=6 #A/mm2
    v_conducteur_rotor = (g*Ps_cu/(p_Cu*(J*1e6)**2))*(1+2*(1/L)*(2*math.pi*R)) #première partie c volume des barre deuxième volume des anneaux de court circuit
    m_conduct_rotor= (8960)*v_conducteur_rotor
    return m_conduct_rotor

def masse_cuivre_accessoires(P_traction,U,c_type,E_batt):
    reseau_servitude = {
    'veryhigh': 25.0,
    'high': 16.0,
    'low': 13.0,
    'PHFCEV_High': 25.0,
    'PHFCEV_Low': 25.0,
    'FCEV': 25.0,
    'Gasoline_PHEV': 20.0,
    'Diesel_MHEV': 20.0,
    'Gasoline_HEV': 20.0,
    'Gasoline_MHEV': 20.0,
    'Gasoline_microHEV': 25.0
    }
    cuivre_inverter = 4*P_traction/U #Formule Excel
    cuivre_cables_voiture = 12*P_traction/U #Formule Excel
    S_cu_info = 0.35*1e-6 #m2 section AWG22 pour bus CAN utilisé par Stellantis dans ses voitures 
    #Bus CAN 2 conducteurs de 0.35 mm2 (AWG22) 
    volume_cable_info_voiture =4000*2*S_cu_info #Donnee de l'ACOM 4km de cable en moyenne en 2025
    cuivre_info =  volume_cable_info_voiture*8960
    cuivre_vehicule_reseau = 2 #2 kg (cable standard 7 m charge 22 kW) cuivre entre OBC et réseau
    cuivre_compteur_borne = 1
    #On compte 1500 cycles complet avant remplacement de la batterie => on compte que la première batterie dans les calculs
    cuivre_prod_transport = (29+55)*1500*E_batt/1000/1000 #29g par MWH produit 55g par MWH transporté 
    cuivre_reseau_servitude = reseau_servitude[c_type]
    dico_masse={
        "cuivre_inverter" :  cuivre_inverter,
        "cuivre_cables_voiture" : cuivre_cables_voiture,
        "cuivre_info" : cuivre_info,
        "cuivre_vehicule_reseau" : cuivre_vehicule_reseau,
        "cuivre_compteur_borne" : cuivre_compteur_borne,
        "cuivre_reseau_servitude" :cuivre_reseau_servitude,
        "cuivre_prod_transport" : cuivre_prod_transport
    }
    return ((cuivre_inverter+cuivre_cables_voiture+cuivre_info+cuivre_vehicule_reseau+cuivre_compteur_borne+cuivre_reseau_servitude),dico_masse)


def volume_ferro_ms():
    # on doit trouver le Diamètre externe du stator à partir du champ dans l'entrefer
    # et des champs statoriques et rotoriques
    # on peut supposer que pour les MSAP Br=Bs
    return True