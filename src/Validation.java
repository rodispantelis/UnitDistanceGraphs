import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Validation {
	
	String filename="E:\\Files\\workspace\\unit\\live_points.csv";
	ArrayList<int[]> coords=new ArrayList<int[]>();
	ArrayList<double[]> coordsd=new ArrayList<double[]>();
	
	int unities=0;
	int errors=0;
	int unities2=0;
	int errors2=0;
	int ns=0, ns2=0;
	int it=400;
	int lines=0;
	int lines2=0;
	int step=1000;

	public static void main(String[] args) {
		Validation v = new Validation();
		v.read();
		v.read2();
		int ii=1;//System.out.println(v.step);
		while(ii*v.step<=v.coords.size()) {
			v.it=ii;
			v.lines2=ii*v.step;
			v.lines=ii*v.step;
			v.validate();
			//v.validate2();
			ii++;
		}
	}
	
	public void read() {
		coords.clear();
		File pr=new File(filename);
		int cnt=0;

		try{
	    	Scanner scanner = new Scanner(pr);
	    	
	    	// Skip header
	    	if (scanner.hasNextLine()) {
	    	    scanner.nextLine();
	    	}
	    	
	    	while(scanner.hasNext() && cnt<(it*step)){
	    		String[] params= scanner.nextLine().split(",");
	    		int n[]= {Integer.parseInt(params[0]),Integer.parseInt(params[1]),Integer.parseInt(params[2]),Integer.parseInt(params[3])};
	    		coords.add(n);
	    		cnt++;
	    	}
	    	scanner.close();
		}
		catch (IOException e) {
		       e.printStackTrace();
		}//System.out.println(coords.size());
	}
	
	public void read2() {
		coordsd.clear();
		File pr=new File(filename);
		int cnt=0;

		try{
	    	Scanner scanner = new Scanner(pr);
	    	
	    	// Skip header
	    	if (scanner.hasNextLine()) {
	    	    scanner.nextLine();
	    	}
	    	
	    	while(scanner.hasNext() && cnt<(it*step)){
	    		String[] params= scanner.nextLine().split(",");
	    		double n[]= {Double.parseDouble(params[4]),Double.parseDouble(params[5])};//,Double.parseDouble(params[2]),Double.parseDouble(params[3])};
	    		coordsd.add(n);
	    		cnt++;
	    	}
	    	scanner.close();
		}
		catch (IOException e) {
		       e.printStackTrace();
		}//System.out.println(coordsd.size());
	}
	
	public void validate() {
		unities=0;
		errors=0;
		ns=0;//System.out.println(lines);
		
		for(int i=0; i<lines;i++){
			for(int k=i+1; k<lines;k++) {
				int s1 = coords.get(i)[0];
				int s2 = coords.get(i)[1];
				int n1 = coords.get(i)[2];
				int n2 = coords.get(i)[3];
				
				int u1 = coords.get(k)[0];
				int u2 = coords.get(k)[1];
				int v1 = coords.get(k)[2];
				int v2 = coords.get(k)[3];
				
				int x_num = s1 * u2 - u1 * s2;
				int x_den = s2 * u2;
				int y_num = n1 * v2 - v1 * n2;
				int y_den = n2 * v2;
				
	            int lhs = (x_num * x_num) * (y_den * y_den) + (y_num * y_num) * (x_den * x_den);
	            int rhs = (x_den * x_den) * (y_den * y_den);
	            
	            if(lhs == rhs) {
	            	unities++;
	            }
	            
	            if(lhs == 0) {
	            	errors++;
	            }
	            ns++;
			}
		}
		
		System.out.println("Nodes: "+lines+", Unit Distance Edges: "+unities
				+" k: "+Math.log(unities)/Math.log(lines)
				+", Errors - duplicate points: "+errors
				//+", Possible edges: "+ns
				);
	}
	

	
	public void validate2() {
		unities2=0;
		errors2=0;
		ns2=0;

		//System.out.println(lines2);
		for(int i=0; i<lines2;i++){
			for(int k=i+1; k<lines2;k++) {
				double s1 = coordsd.get(i)[0];
				double n1 = coordsd.get(i)[1];
				
				double u1 = coordsd.get(k)[0];
				double v1 = coordsd.get(k)[1];

	            
				double dis=Math.sqrt(Math.pow((s1)-(u1), 2)+Math.pow((n1)-(v1), 2));
				
	            if(dis > 0.99999 && dis < 1.00001) {
	            	unities2++;
	            }
	            
	            if(dis == 0) {
	            	errors2++;
	            }
	            ns2++;
			}
		}
		
		System.out.println("Nodes: "+lines2+", Unit Distance Edges: "+unities2
				+" k: "+Math.log(unities2)/Math.log(lines2)
				+", Errors - duplicate points: "+errors2
				//+", Possible edges: "+ns
				);
		}

}
