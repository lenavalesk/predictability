import pandas as pd
import scipy  

%pylab


def logit(p,n):
	n = 2*n
	if isinstance(p, int) or isinstance(p, float) :
		if p == 0:
			p = 1/n
		elif p == 1:
			p = 1-1/n
		logitpred = scipy.log10(p/(1-p))
	else:
		p[p == 0] = 1/n
		p[p == 1] = 1 - (1/n)
		logitpred = scipy.log10(p/(1-p))

	return logitpred

# Load
logFileName = 'predictability2.csv'
log = pd.read_csv(logFileName, encoding='utf-8',  quotechar='"')

log['logit_pred'] = [logit(r.pred, r.CLOZE_nPred) for i,r in log.iterrows()]

log = log[['tId','originales','misWords','pred','logit_pred','CLOZE_nPred']]
log.columns = ['id_text','palabra','palNum','pred','logit_pred','nCompletadas']
log.to_csv('result.csv',index=False)
